# Documentation for `Controllers`

A controller is an implementation of a fragmentation strategy, which decides how particular
ions in an MS1 (survey) scan should be fragmented. The standard fragmentation strategy used
in data-dependant acquisition (DDA) is the Top-N strategy, where the largest N most intense
ion in the survey scan is fragmented. In ViMMS, this strategy is implemented in
[vimms.Controller.topN.TopNController][]. Apart from that, many other DDA strategies have
also been implemented which improve upon the standard TopN controller (for more details,
refer to our papers). 

Additionally, several common data-independent acquisition (DIA) such as
All-ion-fragmentation (AIF), SWATH-MS (Sequential Windowed Acquisition of All Theoretical
Fragment Ion Mass Spectra) have also been implemented as controllers in ViMMS>

The following are broad categories of controllers that are available:

::: vimms.Controller
    handler: python
    rendering:
      show_root_heading: true
      show_source: yes