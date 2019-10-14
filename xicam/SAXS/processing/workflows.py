from xicam.core.execution.workflow import Workflow
from xicam.SAXS.processing.arrayrotate import ArrayRotate
from xicam.SAXS.processing.arraytranspose import ArrayTranspose
from .qintegrate import QIntegratePlugin
from .chiintegrate import ChiIntegratePlugin
from .xintegrate import XIntegratePlugin
from .zintegrate import ZIntegratePlugin
from .cakeintegrate import CakeIntegratePlugin
from .onetime import OneTimeCorrelation
from .fitting import FitScatteringFactor
from .twotime import TwoTimeCorrelation
from .fourierautocorrelator import FourierCorrelation


class ReduceWorkflow(Workflow):
    def __init__(self):
        super(ReduceWorkflow, self).__init__('Reduce')

        self.qintegrate = QIntegratePlugin()
        self.chiintegrate = ChiIntegratePlugin()
        self.xintegrate = XIntegratePlugin()
        self.zintegrate = ZIntegratePlugin()

        self.processes = [self.qintegrate, self.chiintegrate, self.xintegrate, self.zintegrate]
        self.autoConnectAll()


class DisplayWorkflow(Workflow):
    def __init__(self):
        super(DisplayWorkflow, self).__init__('Display')

        self.cake = CakeIntegratePlugin()
        self.processes = [self.cake]
        self.autoConnectAll()


class TwoTime(Workflow):
    name = '2-Time Correlation'

    def __init__(self):
        super(TwoTime, self).__init__()
        self.addProcess(TwoTimeCorrelation())


class OneTime(Workflow):
    name = '1-Time Correlation'

    def __init__(self):
        super(OneTime, self).__init__()
        onetime = OneTimeCorrelation()
        self.addProcess(onetime)
        fitting = FitScatteringFactor()
        self.addProcess(fitting)
        self.autoConnectAll()


class FourierAutocorrelator(Workflow):
    name = 'Fourier Correlation'

    def __init__(self):
        super(FourierAutocorrelator, self).__init__()
        fourier = FourierCorrelation()
        self.addProcess(fourier)



