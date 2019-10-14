import numpy as np
import pyqtgraph as pg
from pyqtgraph.graphicsItems.LegendItem import ItemSample
from qtpy.QtCore import QItemSelection, QPersistentModelIndex, Qt
from qtpy.QtGui import QPen, QStandardItem, QStandardItemModel
from qtpy.QtWidgets import QAbstractItemView, QDialog, QLineEdit, QListView, QTreeView, QVBoxLayout, QWidget


# For some reason, LegendItem.removeItem(ref) wasn't working, so this class stores the data name
class CurveItemSample(ItemSample):
    """
    Provides a custom ItemSample for curve items (PlotItemData) in a plot.
    """
    def __init__(self, item, **kwargs):
        self.name = kwargs.get('name', '')
        super(CurveItemSample, self).__init__(item)


# TODO
# - required:
#   - generic way to display data
#     - do you use toolbar to do this (dropdown of 'keys' you can display)
#     - do you use a context menu
#     - do you add columns to the parent tree items to show g2, ... (probably not best idea, n keys ...)
#   - related; generic way to ignore data
#     - rn, fitcurve is specially handled. how would you do this in a more generic way?
# - nice to have add abilities:
#   - to show symbols
#   - to show lines
# - might be nice to have selection model check/uncheck items
#   - e.g. if selectedItem(s) is checkable: toggle check
#   - model item clicked
class CorrelationView(QWidget):
    """
    Widget for viewing the correlation results.

    This could be generalized into a ComboPlotView / PlotView / ComboView ...
    """
    def __init__(self, model):
        super(CorrelationView, self).__init__()
        self.model = model  # type: QStandardItemModel
        self.resultsList = QTreeView(self)
        self.resultsList.setHeaderHidden(True)
        self.resultsList.setEditTriggers(QAbstractItemView.NoEditTriggers)

        self.resultsList.setSelectionMode(QAbstractItemView.NoSelection)
        self.plotOpts = dict()
        self._plot = pg.PlotWidget(**self.plotOpts)
        self._legend = pg.LegendItem(offset=[-1, 1])
        self.legend.setParentItem(self._plot.getPlotItem())
        self.resultsList.setModel(self.model)
        self.selectionModel = self.resultsList.selectionModel()

        layout = QVBoxLayout()
        layout.addWidget(self.resultsList)
        layout.addWidget(self._plot)
        self.setLayout(layout)

        self.checkedItemIndexes = []
        self._curveItems = []
        self.model.itemChanged.connect(self.updatePlot)

    @property
    def plot(self):
        return self._plot

    @property
    def legend(self):
        return self._legend

    def clearPlot(self):
        self.plot.clear()
        self._clearLegend()
        self._curveItems.clear()

    def _clearLegend(self):
        for curveItem in self._curveItems:
            self.legend.removeItem(name=curveItem.name)
        self.legend.hide()

    def results(self, dataKey):
        for index in self.checkedItemIndexes:
            yield index.data(Qt.UserRole)['data'].get(dataKey, np.array([]))

    def parentItemSet(self):
        parents = set()
        for index in self.checkedItemIndexes:
            parents.add(QPersistentModelIndex(index.parent()))
        return parents

    def updatePlot(self, item: QStandardItem):
        self.clearPlot()

        itemIndex = QPersistentModelIndex(item.index())
        if item.checkState():
            self.checkedItemIndexes.append(itemIndex)
        else:
            # TODO -- might need try for ValueError
            self.checkedItemIndexes.remove(itemIndex)

        g2 = list(self.results('norm-0-g2'))
        g2Err = list(self.results('norm-0-stderr'))
        lagSteps = list(self.results('tau'))
        fitCurve = list(self.results('g2avgFIT1'))
        roiList = list(self.results('dqlist'))

        for roi in range(len(self.checkedItemIndexes)):
            yData = g2[roi].squeeze()
            xData = lagSteps[roi].squeeze()

            color = [float(roi) / len(self.checkedItemIndexes) * 255,
                     (1 - float(roi) / len(self.checkedItemIndexes)) * 255,
                     255]
            self.plotOpts['pen'] = color
            err = g2Err[roi].squeeze()
            self.plot.addItem(pg.ErrorBarItem(x=np.log10(xData), y=yData, top=err, bottom=err, **self.plotOpts))

            curve = self.plot.plot(x=xData, y=yData, **self.plotOpts)
            opts = self.plotOpts.copy()
            opts['pen'] = pg.mkPen(self.plotOpts['pen'])  # type: QPen
            opts['pen'].setStyle(Qt.DashLine)
            name = roiList[roi]
            curveItem = CurveItemSample(curve, name=name)
            self._curveItems.append(curveItem)
            self.legend.addItem(curveItem, curveItem.name)
            if len(fitCurve[roi]) > 0:
                fit_curve = self.plot.plot(x=xData, y=fitCurve[roi].squeeze(), **opts)
                fitCurveItem = CurveItemSample(fit_curve, name=f'{name} (fit)')
                self._curveItems.append(fitCurveItem)
                self.legend.addItem(fitCurveItem, fitCurveItem.name)
            self.legend.show()


class OneTimeView(CorrelationView):
    def __init__(self):
        self.model = QStandardItemModel()
        super(OneTimeView, self).__init__(self.model)
        plotItem = self._plot.getPlotItem()
        plotItem.setLabel('left', 'g<sub>2</sub>(&tau;)', 's')
        plotItem.setLabel('bottom', '&tau;', 's')
        plotItem.setLogMode(x=True)


class TwoTimeView(CorrelationView):
    def __init__(self):
        self.model = QStandardItemModel()
        super(TwoTimeView, self).__init__(self.model)
        plotItem = self._plot.getPlotItem()
        plotItem.setLabel('left', 't<sub>2</sub>', 's')
        plotItem.setLabel('bottom', 't<sub>1</sub>', 's')


class FileSelectionView(QWidget):
    """
    Widget for viewing and selecting the loaded files.
    """
    def __init__(self, headermodel, selectionmodel):
        """

        Parameters
        ----------
        headermodel
            The model to use in the file list view
        selectionmodel
            The selection model to use in the file list view
        """
        super(FileSelectionView, self).__init__()
        # self.parameters = ParameterTree()
        self.fileListView = QListView()
        self.correlationName = QLineEdit()
        self.correlationName.setPlaceholderText('Name of result')

        layout = QVBoxLayout()
        layout.addWidget(self.fileListView)
        layout.addWidget(self.correlationName)
        self.setLayout(layout)

        self.headerModel = headermodel
        self.selectionModel = selectionmodel
        self.fileListView.setModel(headermodel)
        self.fileListView.setSelectionModel(selectionmodel)
        self.fileListView.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.fileListView.setSelectionMode(QAbstractItemView.ExtendedSelection)

        # Make sure when the tabview selection model changes, the file list
        # current item updates
        self.selectionModel.currentChanged.connect(
            lambda current, _:
                self.fileListView.setCurrentIndex(current)
        )

        self.selectionModel.currentChanged.connect(
            lambda current, _:
                self.correlationName.setPlaceholderText(current.data())
        )
