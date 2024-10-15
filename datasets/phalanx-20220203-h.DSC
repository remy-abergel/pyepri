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
XPTS	2000
XMIN	3068.300000
XWID	719.440100
*
*	Documentational Text:
*
TITL	'ref 1 SW 720 TC 163 CT 163 M= 2000 pts'
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
DATE    02/03/22
TIME    21:46:31
CMNT    
SAMP    
SFOR    
STAG    L
EXPT    CW
OXS1    IADC
AXS1    B0VL
AXS2    NONE
AXS3    
A1CT    0.34282
A1SW    0.07198
MWFQ    9.559855e+09
MWPW    9.991e-05
AVGS    1
RESO    TMS_0411
SPTP    0.16385
RCAG    36
RCHM    1
B0MA    0.0003
B0MF    100000
RCPH    0.0
RCOF    0.0
A1RS    2000
RCTC    0.16384
*
************************************************************
*
#DSL	1.0 * DEVICE SPECIFIC LAYER
*

.DVC     acqStart, 1.0


.DVC     fieldCtrl, 1.0

AllegroMode        True
CenterField        3428.20 G
Delay              0.0 s
FieldFlyback       On
FieldWait          Wait LED off
GFactor            2.000000
SetToSampleG       False
StaticFieldMon     3068.300 G
SweepDirection     Up
SweepWidth         719.8 G

.DVC     fieldSweep, 1.0


.DVC     freqCounter, 1.0

FrequencyMon       9.559855 GHz
QMonitBridge       On

.DVC     mwBridge, 1.0

AcqFineTuning      Never
AcqScanFTuning     Off
AcqSliceFTuning    Off
BridgeCalib        100.0
Power              0.09991 mW
PowerAtten         33 dB

.DVC     recorder, 1.0

BaselineCorr       On
NbScansAcc         1
NbScansDone        1
NbScansToDo        1
ReplaceMode        Off
SmoothMode         Auto
SmoothPoints       163

.DVC     scanEnd, 1.0


.DVC     signalChannel, 1.0

AFCTrap            True
AllegroMode        True
AllowShortCt       False
Calibrated         True
ConvTime           163.85 ms
DModDetectSCT      First
DualDetect         OFF
EliDelay           1.0 us
EnableImag         Disable
ExtLockIn          False
ExtTrigger         False
Gain               36 dB
Harmonic           1
HighPass           True
Integrator         False
IsCalibExp         False
ModAmp             3.000 G
ModFreq            100.00 kHz
ModPhase           0.0
Offset             0.0 %
QuadMode           False
Resolution         2000
Resonator          1
SctNorm            False
SctRevision        Allegro
SpuExtension       False
SweepTime          327.70 s
TimeConst          163.84 ms
TimeExp            False
TuneCaps           39

*
************************************************************
