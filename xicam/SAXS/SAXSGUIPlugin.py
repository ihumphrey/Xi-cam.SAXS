import numpy as np
from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *

from xicam.core import msg
from xicam.core.data import load_header, NonDBHeader
from xicam.core.execution.workflow import Workflow

from xicam.plugins import GUIPlugin, GUILayout, manager as pluginmanager

from xicam.gui.widgets.linearworkfloweditor import WorkflowEditor
from xicam.SAXS.processing.workflows import ReduceWorkflow, DisplayWorkflow
from xicam.SAXS.calibration.workflows import SimulateWorkflow
from xicam.SAXS.masking.workflows import MaskingWorkflow
from pyFAI import AzimuthalIntegrator, detectors, calibrant
import pyqtgraph as pg
from functools import partial

from xicam.gui.widgets.tabview import TabView, TabViewSynchronizer

from pyqtgraph.parametertree import ParameterTree, Parameter
from pyqtgraph.parametertree.parameterTypes import ListParameter
from xicam.gui.widgets.imageviewmixins import PolygonROI
from xicam.gui.widgets.ROI import BetterROI
from xicam.SAXS.widgets.SAXSViewerPlugin import SAXSViewerPluginBase
from xicam.SAXS.widgets.views import OneTimeView, TwoTimeView, FileSelectionView, CorrelationView
from xicam.SAXS.processing.workflows import OneTime, TwoTime, FourierAutocorrelator
from xicam.gui import static
import event_model
import dill as pickle
from intake_bluesky.in_memory import BlueskyInMemoryCatalog


class XPCSViewerPlugin(PolygonROI, SAXSViewerPluginBase):
    pass


class XPCSProcessor(ParameterTree):
    def __init__(self, *args, **kwargs):
        super(XPCSProcessor, self).__init__()
        self._paramName = 'Algorithm'
        self._name = 'XPCS Processor'
        self.workflow = None
        self.param = None
        self._workflows = dict()

        self.listParameter = ListParameter(name=self._paramName,
                                           values={'':''},
                                           value='')

        self.param = Parameter(children=[self.listParameter], name=self._name)
        self.setParameters(self.param, showTop=False)

    def update(self, *_):
        for child in self.param.childs[1:]:
            child.remove()

        self.workflow = self._workflows.get(self.listParameter.value().name, self.listParameter.value()())
        self._workflows[self.workflow.name] = self.workflow
        for process in self.workflow.processes:
            self.param.addChild(process.parameter)


class OneTimeProcessor(XPCSProcessor):
    def __init__(self, *args, **kwargs):
        super(OneTimeProcessor, self).__init__()
        self._name = '1-Time Processor'
        self.listParameter.setLimits(OneTimeAlgorithms.algorithms())
        self.listParameter.setValue(OneTimeAlgorithms.algorithms()[OneTimeAlgorithms.default()])

        self.update()
        self.listParameter.sigValueChanged.connect(self.update)


class TwoTimeProcessor(XPCSProcessor):
    def __init__(self, *args, **kwargs):
        super(TwoTimeProcessor, self).__init__()
        self._name = '2-Time Processor'
        self.listParameter.setLimits(TwoTimeAlgorithms.algorithms())
        self.listParameter.setValue(TwoTimeAlgorithms.algorithms()[TwoTimeAlgorithms.default()])

        self.update()
        self.listParameter.sigValueChanged.connect(self.update)


class ProcessingAlgorithms:
    """
    Convenience class to get the available algorithms that can be used for
    one-time and two-time correlations.
    """
    @staticmethod
    def algorithms():
        return {
            TwoTimeAlgorithms.name: TwoTimeAlgorithms.algorithms(),
            OneTimeAlgorithms.name: OneTimeAlgorithms.algorithms()
        }


class TwoTimeAlgorithms(ProcessingAlgorithms):
    name = '2-Time Algorithms'
    @staticmethod
    def algorithms():
        return {TwoTime.name: TwoTime}

    @staticmethod
    def default():
        return TwoTime.name


class OneTimeAlgorithms(ProcessingAlgorithms):
    name = '1-Time Algorithms'
    @staticmethod
    def algorithms():
        return {OneTime.name: OneTime,
                FourierAutocorrelator.name: FourierAutocorrelator}

    @staticmethod
    def default():
        return OneTime.name

class SAXSPlugin(GUIPlugin):
    name = 'SAXS'

    def __init__(self):
        # Late imports required due to plugin system
        from xicam.SAXS.calibration import CalibrationPanel
        from xicam.SAXS.widgets.SAXSMultiViewer import SAXSMultiViewerPlugin
        from xicam.SAXS.widgets.SAXSViewerPlugin import SAXSViewerPluginBase, SAXSCalibrationViewer, SAXSMaskingViewer, \
            SAXSReductionViewer
        from xicam.SAXS.widgets.SAXSToolbar import SAXSToolbar
        from xicam.SAXS.widgets.SAXSSpectra import SAXSSpectra

        self.catalog = BlueskyInMemoryCatalog()
        self.resultsModel = QStandardItemModel()

        # Data model
        self.headermodel = QStandardItemModel()
        self.headerModel = self.headermodel
        self.selectionmodel = QItemSelectionModel(self.headermodel)
        self.selectionModel = self.selectionmodel

        # Initialize workflows
        self.maskingworkflow = MaskingWorkflow()
        self.simulateworkflow = SimulateWorkflow()
        self.displayworkflow = DisplayWorkflow()
        self.reduceworkflow = ReduceWorkflow()

        # Grab the calibration plugin
        self.calibrationsettings = pluginmanager.getPluginByName('xicam.SAXS.calibration',
                                                                 'SettingsPlugin').plugin_object

        # Setup TabViews
        self.calibrationtabview = TabView(self.headermodel, widgetcls=SAXSCalibrationViewer,
                                          selectionmodel=self.selectionmodel,
                                          bindings=[(self.calibrationsettings.sigGeometryChanged, 'setGeometry')],
                                          geometry=self.getAI)
        self.masktabview = TabView(self.headermodel, widgetcls=SAXSMaskingViewer, selectionmodel=self.selectionmodel,
                                   bindings=[('sigTimeChangeFinished', self.indexChanged),
                                             (self.calibrationsettings.sigGeometryChanged, 'setGeometry')],
                                   geometry=self.getAI)
        self.reducetabview = TabView(self.headermodel, widgetcls=SAXSReductionViewer,
                                     selectionmodel=self.selectionmodel,
                                     bindings=[('sigTimeChangeFinished', self.indexChanged),
                                               (self.calibrationsettings.sigGeometryChanged, 'setGeometry')],
                                     geometry=self.getAI)
        self.comparemultiview = SAXSMultiViewerPlugin(self.headermodel, self.selectionmodel)

        # self.tabviewsynchronizer = TabViewSynchronizer(
        #     [self.calibrationtabview, self.masktabview, self.reducetabview, self.comparemultiview.leftTabView])

        # Setup toolbars
        self.toolbar = SAXSToolbar(self.headermodel, self.selectionmodel)
        self.calibrationtabview.kwargs['toolbar'] = self.toolbar
        self.reducetabview.kwargs['toolbar'] = self.toolbar

        # Setup calibration widgets
        self.calibrationsettings.setModels(self.headermodel, self.calibrationtabview.selectionmodel)
        self.calibrationpanel = CalibrationPanel(self.headermodel, self.calibrationtabview.selectionmodel)
        self.calibrationpanel.sigDoCalibrateWorkflow.connect(self.doCalibrateWorkflow)
        self.calibrationsettings.sigGeometryChanged.connect(self.doSimulateWorkflow)

        # Setup masking widgets
        self.maskeditor = WorkflowEditor(self.maskingworkflow)
        self.maskeditor.sigWorkflowChanged.connect(self.doMaskingWorkflow)

        # Setup reduction widgets
        self.displayeditor = WorkflowEditor(self.displayworkflow)
        self.reduceeditor = WorkflowEditor(self.reduceworkflow)
        self.reduceplot = SAXSSpectra(self.reduceworkflow, self.toolbar)
        self.toolbar.sigDoWorkflow.connect(partial(self.doReduceWorkflow))
        self.reduceeditor.sigWorkflowChanged.connect(self.doReduceWorkflow)
        self.displayeditor.sigWorkflowChanged.connect(self.doDisplayWorkflow)
        self.reducetabview.currentChanged.connect(self.headerChanged)
        self.reducetabview.currentChanged.connect(self.headerChanged)

        # Setup correlation views
        self.twoTimeView = TwoTimeView()
        self.twoTimeFileSelection = FileSelectionView(self.headerModel, self.selectionModel)
        self.twoTimeProcessor = TwoTimeProcessor()
        self.twoTimeToolBar = QToolBar()
        self.twoTimeToolBar.addAction(QIcon(static.path('icons/run.png')), 'Process', self.processTwoTime)
        self.twoTimeToolBar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        self.oneTimeView = OneTimeView()
        self.oneTimeFileSelection = FileSelectionView(self.headerModel, self.selectionModel)
        self.oneTimeProcessor = OneTimeProcessor()
        self.oneTimeToolBar = QToolBar()
        self.oneTimeToolBar.addAction(QIcon(static.path('icons/run.png')), 'Process', self.processOneTime)
        self.oneTimeToolBar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        # Setup more bindings
        self.calibrationsettings.sigSimulateCalibrant.connect(partial(self.doSimulateWorkflow))

        self.stages = {
            'Calibrate': GUILayout(self.calibrationtabview,
                                   # pluginmanager.getPluginByName('SAXSViewerPlugin', 'WidgetPlugin').plugin_object()
                                   right=self.calibrationsettings.widget,
                                   rightbottom=self.calibrationpanel,
                                   top=self.toolbar),
            'Mask': GUILayout(self.masktabview,
                              right=self.maskeditor,
                              top=self.toolbar),
            'Reduce': GUILayout(self.reducetabview,
                                bottom=self.reduceplot, right=self.reduceeditor, righttop=self.displayeditor,
                                top=self.toolbar),
            # 'Compare': GUILayout(self.comparemultiview, top=self.toolbar, bottom=self.reduceplot,
            #                      right=self.reduceeditor),
            '2-Time Correlation': GUILayout(self.twoTimeView,
                                            top=self.twoTimeToolBar,
                                            right=self.twoTimeFileSelection,
                                            rightbottom=self.twoTimeProcessor, ),
            '1-Time Correlation': GUILayout(self.oneTimeView,
                                            top=self.oneTimeToolBar,
                                            right=self.oneTimeFileSelection,
                                            rightbottom=self.oneTimeProcessor, )
        }
        self._results = []

        super(SAXSPlugin, self).__init__()

        # Start visualizations
        self.displayworkflow.visualize(self.reduceplot, imageview=lambda: self.reducetabview.currentWidget(),
                                       toolbar=self.toolbar)

    # def experimentChanged(self):
    #     self.doReduceWorkflow(self.reduceworkflow)

    def getAI(self):
        """ Convenience method to get current field's AI """
        device = self.toolbar.detectorcombobox.currentText()
        ai = self.calibrationsettings.AI(device)
        return ai

    def indexChanged(self):
        if not self.reduceplot.toolbar.multiplot.isChecked():
            self.doReduceWorkflow(self.reduceworkflow)

    def headerChanged(self):
        self.toolbar.updatedetectorcombobox(None, None)
        self.doReduceWorkflow()
        self.doDisplayWorkflow()

    def appendHeader(self, header: NonDBHeader, **kwargs):
        item = QStandardItem(header.startdoc.get('sample_name', '????'))
        item.header = header
        self.headermodel.appendRow(item)
        self.selectionmodel.setCurrentIndex(self.headermodel.index(self.headermodel.rowCount() - 1, 0),
                                            QItemSelectionModel.Rows)
        self.headermodel.dataChanged.emit(QModelIndex(), QModelIndex())

        # Load any reduced (processed) data
        reduced = False
        for descriptor in header.descriptordocs:
            if descriptor['name'] == '1-Time':
                reduced = True
                break
        paths = header.startdoc.get('paths')
        for path in paths:
            if reduced:
                startItem = QStandardItem(header.startdoc.get('sample_name', '??'))
                eventlist = header.eventdocs
                for event in eventlist:
                    eventItem = QStandardItem(repr(event['data']['dqlist']))
                    eventItem.setData(event, Qt.UserRole)
                    eventItem.setCheckable(True)
                    startItem.appendRow(eventItem)
                # TODO -- properly add to view (one-time or 2-time, etc.)
                self.oneTimeView.model.invisibleRootItem().appendRow(startItem)

    def currentheader(self):
        return self.headerModel.itemFromIndex(self.selectionModel.currentIndex()).header

    def currentheaders(self):
        selected_indices = self.selectionModel.selectedIndexes()
        headers = []
        for model_index in selected_indices:
            headers.append(self.headerModel.itemFromIndex(model_index).header)
        return headers

    def doCalibrateWorkflow(self, workflow: Workflow):
        data = self.calibrationtabview.currentWidget().header.meta_array()[0]
        device = self.toolbar.detectorcombobox.currentText()
        ai = self.calibrationsettings.AI(device)
        # ai.detector = detectors.Pilatus2M()
        calibrant = self.calibrationpanel.parameter['Calibrant Material']

        def setAI(result):
            self.calibrationsettings.setAI(result['ai'].value, device)
            self.doMaskingWorkflow()

        workflow.execute(None, data=data, ai=ai, calibrant=calibrant, callback_slot=setAI, threadkey='calibrate')

    def doSimulateWorkflow(self):
        # TEMPORARY HACK for demonstration
        if self.reducetabview.currentWidget():
            self.reducetabview.currentWidget().setTransform()

        if not self.calibrationtabview.currentWidget(): return
        data = self.calibrationtabview.currentWidget().header.meta_array()[0]
        device = self.toolbar.detectorcombobox.currentText()
        ai = self.calibrationsettings.AI(device)
        calibrant = self.calibrationpanel.parameter['Calibrant Material']
        outputwidget = self.calibrationtabview.currentWidget()

        def showSimulatedCalibrant(result=None):
            outputwidget.setCalibrantImage(result['data'].value)

        self.simulateworkflow.execute(None, data=data, ai=ai, calibrant=calibrant, callback_slot=showSimulatedCalibrant,
                                      threadkey='simulate')

    def doMaskingWorkflow(self, workflow=None):
        if not self.masktabview.currentWidget(): return
        if not self.checkPolygonsSet(self.maskingworkflow):
            data = self.masktabview.currentWidget().header.meta_array()[0]
            device = self.toolbar.detectorcombobox.currentText()
            ai = self.calibrationsettings.AI(device)
            outputwidget = self.masktabview.currentWidget()

            def showMask(result=None):
                if result:
                    outputwidget.setMaskImage(result['mask'].value)
                else:
                    outputwidget.setMaskImage(None)
                self.doDisplayWorkflow()
                self.doReduceWorkflow()

            if not workflow: workflow = self.maskingworkflow
            workflow.execute(None, data=data, ai=ai, callback_slot=showMask, threadkey='masking')

    def doDisplayWorkflow(self):
        if not self.reducetabview.currentWidget(): return
        currentwidget = self.reducetabview.currentWidget()
        data = currentwidget.header.meta_array()[currentwidget.timeIndex(currentwidget.timeLine)[0]]
        device = self.toolbar.detectorcombobox.currentText()
        ai = self.calibrationsettings.AI(device)
        mask = self.maskingworkflow.lastresult[0]['mask'].value if self.maskingworkflow.lastresult else None
        outputwidget = currentwidget

        def showDisplay(*results):
            outputwidget.setResults(results)

        self.displayworkflow.execute(None, data=data, ai=ai, mask=mask, callback_slot=showDisplay, threadkey='display')

    def doReduceWorkflow(self):
        if not self.reducetabview.currentWidget(): return
        multimode = self.reduceplot.toolbar.multiplot.isChecked()
        currentwidget = self.reducetabview.currentWidget()
        data = currentwidget.header.meta_array()
        if not multimode:
            data = [data[currentwidget.timeIndex(currentwidget.timeLine)[0]]]
        device = self.toolbar.detectorcombobox.currentText()
        ai = self.calibrationsettings.AI(device)
        ai = [ai] * len(data)
        mask = [self.maskingworkflow.lastresult[0]['mask'].value if self.maskingworkflow.lastresult else None] * len(
            data)
        outputwidget = self.reduceplot

        # outputwidget.clear_all()

        def showReduce(*results):
            self.reduceplot.plot_mode(results)
            pass

        self.reduceworkflow.execute_all(None, data=data, ai=ai, mask=mask, callback_slot=showReduce, threadkey='reduce')

    def checkPolygonsSet(self, workflow: Workflow):
        """
        Check for any unset polygonmask processes; start masking mode if found

        Parameters
        ----------
        workflow: Workflow

        Returns
        -------
        bool
            True if unset polygonmask process is found

        """
        pluginmaskclass = pluginmanager.getPluginByName('Polygon Mask', 'ProcessingPlugin')
        for process in workflow.processes:
            if isinstance(process, pluginmaskclass.plugin_object):
                if process.polygon.value is None:
                    self.startPolygonMasking(process)
                    return True
        return False

    def startPolygonMasking(self, process):
        self.setEnabledOuterWidgets(False)

        # Start drawing mode
        viewer = self.masktabview.currentWidget()  # type: SAXSViewerPluginBase
        viewer.imageItem.setDrawKernel(kernel=np.array([[0]]), mask=None, center=(0, 0), mode='add')
        viewer.imageItem.drawMode = self.drawEvent
        viewer.maskROI.clearPoints()

        # Setup other signals
        process.parameter.child('Finish Mask').sigActivated.connect(partial(self.finishMask, process))
        process.parameter.child('Clear Selection').sigActivated.connect(self.clearMask)

    def setEnabledOuterWidgets(self, enabled):
        # Disable other widgets
        mainwindow = self.masktabview.window()
        for dockwidget in mainwindow.findChildren(QDockWidget):
            dockwidget.setEnabled(enabled)
        mainwindow.rightwidget.setEnabled(True)
        self.maskeditor.workflowview.setEnabled(enabled)
        self.masktabview.tabBar().setEnabled(enabled)
        mainwindow.menuBar().setEnabled(enabled)
        mainwindow.pluginmodewidget.setEnabled(enabled)

    def clearMask(self):
        viewer = self.masktabview.currentWidget()  # type: SAXSViewerPluginBase
        viewer.maskROI.clearPoints()

    def finishMask(self, process, sender):
        viewer = self.masktabview.currentWidget()  # type: SAXSViewerPluginBase
        process.polygon.value = np.array([list(handle['pos']) for handle in viewer.maskROI.handles])
        self.setEnabledOuterWidgets(True)

        # End drawing mode
        viewer.imageItem.drawKernel = None
        viewer.maskROI.clearPoints()
        process.parameter.clearChildren()

        # Redo workflow with polygon
        self.doMaskingWorkflow()

    def drawEvent(self, kernel, imgdata, mask, ss, ts, event):
        viewer = self.masktabview.currentWidget()  # type: SAXSViewerPluginBase
        viewer.maskROI.addFreeHandle(viewer.view.vb.mapSceneToView(event.scenePos()))
        if len(viewer.maskROI.handles) > 1:
            viewer.maskROI.addSegment(viewer.maskROI.handles[-2]['item'], viewer.maskROI.handles[-1]['item'])

    def processOneTime(self):
        self.process(self.oneTimeProcessor,
                     callback_slot=partial(self.saveResult, fileSelectionView=self.oneTimeFileSelection),
                     finished_slot=partial(self.createDocument, view=self.oneTimeView))

    def processTwoTime(self):
        self.process(self.twoTimeProcessor,
                     callback_slot=partial(self.saveResult, fileSelectionView=self.twoTimeFileSelection),
                     finished_slot=partial(self.createDocument, view=self.twoTimeView))

    def process(self, processor: XPCSProcessor, **kwargs):
        if processor:
            workflow = processor.workflow

            data = [header.meta_array() for header in self.currentheaders()]
            # currentWidget = self.rawTabView.currentWidget()
            currentWidget = self.reducetabview.currentWidget()
            rois = [item for item in currentWidget.view.items if isinstance(item, BetterROI)]
            labels = [currentWidget.poly_mask()] * len(data)
            numLevels = [1] * len(data)

            numBufs = []
            for i, _ in enumerate(data):
                shape = data[i].shape[0]
                # multi_tau_corr requires num_bufs to be even
                if shape % 2:
                    shape += 1
                numBufs.append(shape)

            if kwargs.get('callback_slot'):
                callbackSlot = kwargs['callback_slot']
            else:
                callbackSlot = self.saveResult
            if kwargs.get('finished_slot'):
                finishedSlot = kwargs['finished_slot']
            else:
                finishedSlot = self.createDocument

            workflowPickle = pickle.dumps(workflow)
            workflow.execute_all(None,
                                 data=data,
                                 labels=labels,
                                 num_levels=numLevels,
                                 num_bufs=numBufs,
                                 callback_slot=callbackSlot,
                                 finished_slot=partial(finishedSlot,
                                                       header=self.currentheader(),
                                                       roi=rois[0],  # todo -- handle multiple rois
                                                       workflow=workflowPickle))
            # TODO -- should header be passed to callback_slot
            # (callback slot handle can handle multiple data items in data list)

    def saveResult(self, result, fileSelectionView=None):
        if fileSelectionView:
            analyzed_results = dict()

            if not fileSelectionView.correlationName.displayText():
                analyzed_results['result_name'] = fileSelectionView.correlationName.placeholderText()
            else:
                analyzed_results['result_name'] = fileSelectionView.correlationName.displayText()
            analyzed_results = {**analyzed_results, **result}

            self._results.append(analyzed_results)

    def createDocument(self, view: CorrelationView, header, roi, workflow):

        # TODO -- remove this temp code for time time
        if type(view) is TwoTimeView:
            import pyqtgraph as pg
            from xicam.gui.widgets.imageviewmixins import LogScaleIntensity
            # why multiple results?
            g2 = self._results[0]['g2'].value.squeeze()
            img = LogScaleIntensity()
            img.setImage(g2)
            img.show()
        ###

        self.catalog.upsert(self._createDocument, (self._results, header, roi, workflow), {})
        # TODO -- make sure that this works for multiple selected series to process
        key = list(self.catalog)[-1]

        # TODO -- make sure 'result_name' is unique in model
        parentItem = QStandardItem(self._results[-1]['result_name'])
        for name, doc in self.catalog[key].read_canonical():
            if name == 'event':
                resultsModel = view.model
                # item = QStandardItem(doc['data']['name'])
                item = QStandardItem(repr(doc['data']['dqlist']))
                item.setData(doc, Qt.UserRole)
                item.setCheckable(True)
                parentItem.appendRow(item)
                selectionModel = view.selectionModel
                selectionModel.reset()
                selectionModel.setCurrentIndex(
                    resultsModel.index(resultsModel.rowCount() - 1, 0), QItemSelectionModel.Rows)
                selectionModel.select(selectionModel.currentIndex(), QItemSelectionModel.SelectCurrent)
        resultsModel.appendRow(parentItem)
        self._results = []

    def _createDocument(self, results, header, roi, workflow):
        import time
        timestamp = time.time()

        run_bundle = event_model.compose_run()
        yield 'start', run_bundle.start_doc

        # TODO -- make sure workflow pickles, or try dill / cloudpickle
        source = 'Xi-cam'

        peek_result = results[0]
        g2_shape = peek_result['g2'].value.shape[0]
        # TODO -- make sure norm-0-stderr is calculated and added to internal process documents
        import numpy as np
        g2_err = np.zeros(g2_shape)
        g2_err_shape = g2_shape
        tau_shape = peek_result['lag_steps'].value.shape[0]
        workflow = []
        workflow_shape = len(workflow)

        reduced_data_keys = {
            'norm-0-g2': {'source': source, 'dtype': 'number', 'shape': [g2_shape]},
            'norm-0-stderr': {'source': source, 'dtype': 'number', 'shape': [g2_err_shape]},
            'tau': {'source': source, 'dtype': 'number', 'shape': [tau_shape]},
            'g2avgFIT1': {'source': source, 'dtype': 'number', 'shape': [tau_shape]},
            'dqlist': {'source': source, 'dtype': 'string', 'shape': []},  # todo -- shape
            'workflow': {'source': source, 'dtype': 'string', 'shape': [workflow_shape]}
        }
        reduced_stream_name = 'reduced'
        reduced_stream_bundle = run_bundle.compose_descriptor(data_keys=reduced_data_keys,
                                                              name=reduced_stream_name)
        yield 'descriptor', reduced_stream_bundle.descriptor_doc

        # todo -- peek frame shape
        frame_data_keys = {'frame': {'source': source, 'dtype': 'number', 'shape': []}}
        frame_stream_name = 'primary'
        frame_stream_bundle = run_bundle.compose_descriptor(data_keys=frame_data_keys,
                                                            name=frame_stream_name)
        yield 'descriptor', frame_stream_bundle.descriptor_doc

        # todo -- store only paths? store the image data itself (memory...)
        # frames = header.startdoc['paths']
        frames = []
        for frame in frames:
            yield 'event', frame_stream_bundle.compose_event(
                data={frame},
                timestamps={timestamp}
            )

        for result in results:
            yield 'event', reduced_stream_bundle.compose_event(
                data={'norm-0-g2': result['g2'].value,
                      'norm-0-stderr': g2_err,
                      'tau': result['lag_steps'].value,
                      'g2avgFIT1': result['fit_curve'].value,
                      'dqlist': roi,
                      'workflow': workflow},
                timestamps={'norm-0-g2': timestamp,
                            'norm-0-stderr': timestamp,
                            'tau': timestamp,
                            'g2avgFIT1': timestamp,
                            'dqlist': timestamp,
                            'workflow': workflow}
            )

        yield 'stop', run_bundle.compose_stop()
