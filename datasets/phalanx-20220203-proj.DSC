#DESC	1.2 * DESCRIPTOR INFORMATION ***********************
*
*	Dataset Type and Format:
*
DSRC	EXP
BSEQ	BIG
IKKF	REAL
XTYP	IDX
YTYP	IGD
ZTYP	NODATA
*
*	Item Formats:
*
IRFMT	D
YFMT	D
*
*	Data Ranges and Resolutions:
*
XPTS	2000
XMIN	3068.300000
XWID	719.440100
YPTS	113
YMIN	0.796460
YWID	178.407080
*
*	Documentational Text:
*
TITL	'Img ZY SW 720 TC 163 CT 163 M= 2000 pts'
IRNAM	'Intensity'
XNAM	'Field'
YNAM	'Angle'
IRUNI	''
XUNI	'G'
YUNI	''
*
************************************************************
*
#SPL	1.2 * STANDARD PARAMETER LAYER
*
OPER    irpe
DATE    02/04/22
TIME    08:55:56
CMNT    
SAMP    
SFOR    
STAG    L
EXPT    CWIMG
OXS1    IADC
AXS1    B0VL
AXS2    GRAD
AXS3    
A1CT    0.34282
A1SW    0.07198
MWFQ    9.559145e+09
IMGP    0.000
IMGT    -90.000
GRAD    168.000
IMTY    2D
MWPW    9.988e-05
A2RS    113
A2LL    1
A2SW    1
AVGS    1
RESO    TMS_0411
SPTP    0.16384
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
StaticFieldMon     3080.537 G
SweepDirection     Up
SweepWidth         719.8 G

.DVC     fieldSweep, 1.0


.DVC     freqCounter, 1.0

FrequencyMon       9.559145 GHz
QMonitBridge       On

.DVC     grdUnit, 1.0

AnglePhi           0.000 deg
AnglePsi           168.0000 G/cm
AngleTheta         -90.000 deg
BandWidth          3.500 G
CurrentCheck       Wait stable
FieldOfView        30.00 mm
FirstAlpha         0.796 deg
ImageOrient        ZY
ImageType          2D
MaxPsi             0.00 deg
MinSW              719.76 G
NPixel             144
NrOfAlpha          113
NrOfBeta           1
NrOfProfs          113
NrOfPsi            1
PixelSize          0.2083 mm
SWTime             327.680 s
SleepSecs          0.0 s
SpecWidth          3.500 G
SpectralRes        0.02431 G/p
TotalTime          621.5 min

.DVC     mwBridge, 1.0

AcqFineTuning      Never
AcqScanFTuning     Off
AcqSliceFTuning    Off
BridgeCalib        100.0
Power              0.09988 mW
PowerAtten         33 dB

.DVC     ramp2, 1.0

Flyback            False
NbPoints           113
StartValue         1
SweepData          
SweepWidth         1
sweepType          Ramp up

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
ConvTime           163.84 ms
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
SweepTime          327.68 s
TimeConst          163.84 ms
TimeExp            False
TuneCaps           39

*
************************************************************
