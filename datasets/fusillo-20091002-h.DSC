#DESC	1.2 * DESCRIPTOR INFORMATION ***********************
*
*	Dataset Type and Format:
*
DSRC	EXP
BSEQ	BIG
IKKF	REAL
XTYP	IDX
YTYP	NODATA
ZTYP	NODATA
*
*	Item Formats:
*
IRFMT	D
*
*	Data Ranges and Resolutions:
*
XPTS	500
XMIN	333.450000
XWID	132.235000
*
*	Documentational Text:
*
TITL	'ref_3D_ZX_fusilli_res=0.5mm_sweep=1.25s_500pts_TEMPO=4mM'
IRNAM	'Intensity'
XNAM	'Field'
IRUNI	''
XUNI	'G'
*
************************************************************
*
#SPL	1.2 * STANDARD PARAMETER LAYER
*
OPER    irpe
DATE    10/02/09
TIME    17:08:00
CMNT    
SAMP    
SFOR    
STAG    C
EXPT    CW
OXS1    IADC
AXS1    B0VL
AXS2    NONE
AXS3    
A1CT    0.03997
A1SW    0.01325
MWFQ    1.086258e+09
MWPW    0.02005
AVGS    1
RESO    BLGR_23_06
SPTP    0.0025
RCAG    60
RCHM    1
B0MA    6e-05
B0MF    100000
RCPH    180.0
RCOF    0.0
A1RS    1024
RCTC    0.00256
*
************************************************************
*
#DSL	1.0 * DEVICE SPECIFIC LAYER
*

.DVC     acqStart, 1.0


.DVC     fieldCtrl, 1.0

AllegroMode        True
CenterField        399.70 G
Delay              0.0 s
FieldFlyback       On
FieldWait          Wait LED off
GFactor            2.000000
SetToSampleG       False
StaticFieldMon     333.450 G
SweepDirection     Up
SweepWidth         132.5 G

.DVC     fieldSweep, 1.0


.DVC     freqCounter, 1.0

FrequencyMon       1.086258 GHz
QMonitBridge       On

.DVC     mwBridge, 1.0

AcqFineTuning      Never
Power              20.05 mW
PowerAtten         13.00 dB

.DVC     recorder, 1.0

BaselineCorr       On
NbScansAcc         1
NbScansDone        1
NbScansToDo        1
ReplaceMode        Off
SmoothMode         Auto
SmoothPoints       1

.DVC     scanEnd, 1.0


.DVC     signalChannel, 1.0

AFCTrap            True
AllegroMode        True
Calibrated         True
DModDetectSCT      First
DualDetect         OFF
EliDelay           1.0 us
EnableImag         Disable
ExtLockIn          False
ExtTrigger         False
Gain               60 dB
GateTime           2.50 ms
Harmonic           1
HighPass           True
Integrator         False
ModAmp             0.60 G
ModFreq            100.00 kHz
ModPhase           180.0
NbPoints           500
Offset             0.0 %
QuadMode           False
Resonator          2
SamplingTime       2.50 ms
ScanTime           1.25 s
SctNorm            False
SctRevision        Allegro
TimeConst          2.56 ms
TimeExp            False
TuneCaps           94

*
************************************************************
