from SignalObject import *
import numpy as np


def init():
    pass

def case1(signal1):
    print('Case 1: Show Signal Object:')
    signal1.showInfo()
    signal1.plotSignal()

def case2(signal1, alpha):
    print('Case 2: alpha multiply with this signal: ')
    signal1.showInfo()
    print('Alpha: ', alpha)
    signal1.mulAlpha(alpha)
    print(signal1.getAmplitude())
    signal1.plotSignal()

def case3(signal1, signal2):
    print('Case 3: Calculate sum of two signals: ')
    signal1.showInfo()
    signal2.showInfo()
    signal1.plotSignal()
    signal2.plotSignal()
    signalsum = signal1.add(signal2)
    print(signalsum.getAmplitude())
    print(signalsum.getStartPos())
    signalsum.plotSignal()

def case4():
    print('Case 4: Time shifting: ')
    signaltime = Signal(np.array([0]), 0, 44100, 'Sample Signal timeshifting', 1)
    c = np.array([1, 0.75, 0.5, 0.25])
    signaltime.getRandSignal(len(c), c, 0)
    print('before: ')
    signal1.showInfo()
    signaltime.plotSignal()
    print('the startpos before: ', signaltime.getStartPos())
    n = int(input('Press the time you want to shift: '))

    signaltimeres = signaltime.timeShifting(n)
    signaltimeres.plotSignal()
    print('the startpos after shifting: ', n, signaltimeres.getStartPos())

def case5(signal3):
    print('Case 5: Time inversed of Signal: ')
    signal1.showInfo()
    signal3.plotSignal()
    print(signal3.getAmplitude())
    signal3.inverse()
    print('Signal after inversed: ')
    print(signal3.getAmplitude())
    signal3.plotSignal()

def case6(signal1, signal2):
    print('Case 6: CrossCorrelation of two signals')
    signal1.showInfo()
    signal2.showInfo()
    signalres1 = signal1.convolveUsingDefineMethod(signal2)
    print(signalres1.getAmplitude())
    print(signalres1.getStartPos())

    # signalres2= signal1.crossCorrelationUsingDefineMethod(signal2)
    # print(signalres2.getAmplitude())
    # print(signalres2.getStartPos())

def case7():
    print('Case 7: Convolve of two signals')
    e = np.array([1, 3, 4])
    f = np.array([2, -1, 5])
    signal3 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 3', 1)
    signal4 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 4', 1)
    signal3.getRandSignal(len(e), e, 2)
    signal4.getRandSignal(len(f), f, 3)
    signal3.showInfo()
    signal4.showInfo()
    signalres = signal3.convolve(signal4)
    signalres.plotSignal()
    print(signalres.getAmplitude())

def case8(signal):
    print('Case 8: Calculate Energy and Wattage: ')
    signal.showInfo()
    print('Energy: ',signal.getEnergy())
    print('Wattage: ',signal.getWattage())

def case9():
    print('Case 9: Transform DFT: ')
    signal4 = Signal(np.array([1, 2, 3, 0]), 0, 41000, 'Signal test dft', 4)
    signal4.showInfo()
    signal4.showDft()
    signal4.showIdft([2, 2+2j,-2,2-2j])

def case10():
    print('Case 10: Transform FFT: ')
    signal4 = Signal(np.array([1, 2, 3, 0]), 0, 41000, 'Signal test fft', 4)
    signal4.showInfo()
    signal4.showFft()
    signal4.showIfft([2,2+2j,-2, 2-2j])
    print(signal4.fftUsingDefineMethod())

def case11():
    print('Case 11: Types of Object: ')
    signaltype = Signal(np.array([0]), 0, 44100, 'Sample Signal types', 1)
    d = np.array([1, 2, 0, 0, 0, 0])
    signaltype.getRandSignal(len(d), d, -2)
    signaltype.showInfo()
    print(signaltype.isCausalSignal())
    print(signaltype.isAntiCausalSignal())
    print(signaltype.isNonCausalSignal())

if __name__ == '__main__':
    signal1 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 1', 1)
    signal2 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 2', 1)
    signal3 = Signal(np.array([0]), 0, 44100, 'Sample Signal Object 3', 1)
    a = np.array([-1, 2, -5, 0, -4, 4, -4])
    b = np.array([3, -1, 1, 0, 1, 5, 7])
    pos1 = 2
    pos2 = 3
    pos3 = 3
    signal1.getRandSignal(len(a), a, pos1)
    signal2.getRandSignal(len(b), b, pos2)
    signal3.getRandSignal(len(b), b, pos3)
    print('Started Program')
    while True:
        print('Case 1: Plot Signal\nCase 2: Multiply with Alpha\nCase 3: Sum of two Signals\nCase 4: Time Shifting\nCase 5: Inverse Signals with time\nCase 6: CrossCorrelation of two signals\nCase 7: Convolve of two signals\nCase 8: Calculate Energy and Wattage\nCase 9: Transform DFT\nCase 10: Transform FFT\nCase 11: Types of Object')
        n = int(input('Press your select: '))
        if n == 0:
            print('Ended Program!!!')
        if n == 1:
            case1(signal1)
        if n == 2:
            alpha = int(input('Press Your Alpha: '))
            case2(signal1, alpha)
        if n == 3:
            case3(signal1, signal2)
        if n == 4:
            case4()
        if n == 5:
            case5(signal3)
        if n == 6:
            case6(signal1, signal2)
        if n == 7:
            case7()
        if n == 8:
            case8(signal1)
        if n == 9:
            case9()
        if n == 10:
            case10()
        if n == 11:
            case11()
