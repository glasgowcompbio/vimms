---
nav_exclude: true
---
# Documentation for `Controllers`

Controllers in ViMMS implement fragmentation strategies, determining which ions in an MS1 (survey) scan should be fragmented. A standard fragmentation strategy used in data-dependant acquisition (DDA) is the Top-N strategy, where the top N most intense ions in the survey scan are fragmented. This strategy is implemented in the [TopNController](https://github.com/glasgowcompbio/vimms/tree/master/vimms/Controller/topN/TopNController) in ViMMS.

In addition to Top-N, several other DDA strategies have been implemented that improve upon the standard TopN controller (for more details, refer to our papers).

ViMMS also includes implementations of several common data-independent acquisition (DIA) strategies such as All-ion-fragmentation (AIF), SWATH-MS (Sequential Windowed Acquisition of All Theoretical Fragment Ion Mass Spectra).

The following are broad categories of controllers that are available in ViMMS:

::: vimms.Controller
    handler: python
    rendering:
      show_root_heading: true
      show_source: yes