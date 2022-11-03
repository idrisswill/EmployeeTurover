VERSION = '1.0.0'

# ---- Default notebook name ---------------------------------------
#
DEFAULT_NOTEBOOK_NAME = "EMPLOYEETURNOVER"

# ---- Styles ------------------------------------------------------
#
PROJECT_MPLSTYLE = 'core/mplstyles/custom.mplstyle'
PROJECT_CSSFILE  = 'core/css/custom.css'

# ---- Save figs or not (yes|no)
#      Overided by env : TWEET_SAVE_FIGS
#      
SAVE_FIGS    = False

# ---- Catalog file, a json description of all notebooks ------------
#
CATALOG_FILE    = '../core/logs/catalog.json'
PROFILE_FILE    = '../core/ci/default.yml'

# ---- CI report files ----------------------------------------------
#
CI_REPORT_JSON = '../core/logs/ci_report.json'
CI_REPORT_HTML = '../core/logs/ci_report.html'
CI_ERROR_FILE  = '../core/logs/ci_ERROR.txt'

# ---- Used modules -------------------------------------------------
#
USED_MODULES   = ['tensorflow', 'tensorflow.keras', 'numpy', 'sklearn',
                  'skimage', 'matplotlib', 'plotly', 'pandas', 'jupyterlab',
                  'TensorBoard', 'torchvision']

# -------------------------------------------------------------------
