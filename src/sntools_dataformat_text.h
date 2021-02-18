
#define OPTMASK_TEXT_HEAD 2  // matches snana.car
#define OPTMASK_TEXT_OBS  4
#define OPTMASK_TEXT_SPEC 8

#define MSKOPT_PARSE_TEXT_FILE  MSKOPT_PARSE_WORDS_FILE + MSKOPT_PARSE_WORDS_IGNORECOMMENT

#define MXVAROBS_TEXT 20
struct {
  int MJD, BAND, FIELD, FLUXCAL, FLUXCALERR ;
  int ZPFLUX, ZPERR, PSF, SKYSIG, SKYSIG_T, SKYSIG_GAIN ;
  int GAIN, PHOTFLAG, PHOTPROB, XPIX, YPIX, CCDNUM; 
  int SIMEPOCH_MAG ; // EPFILTREST, EPMAGREST??
} IVAROBS_TEXT ;

struct {
  int LAMMIN, LAMMAX, LAMAVG, FLAM, FLAMERR;
  int SIM_GENFLAM, SIM_GENMAG ;
} IVARSPEC_TEXT ;

struct {
  int  NVERSION ;
  char DATA_PATH[MXPATHLEN];   
  char PHOT_VERSION[MXPATHLEN];   
  char LIST_FILE[MXPATHLEN] ;
  char README_FILE[MXPATHLEN];

  int  NFILE;
  char **DATA_FILE_LIST ;

} TEXT_VERSION_INFO ;


struct {
  int IPTR_READ ;  // pointer to current word in file
  int NWD_TOT ;   // total number of words in file
  int NVAROBS ;   // number of variables following each OBS key
  int NVARSPEC ;  // number of spec variables

  char VARNAME_OBS_LIST[MXVAROBS_TEXT][32] ;   // OBS-column names
  char VARNAME_SPEC_LIST[MXVAROBS_TEXT][32] ;  // SPEC-column names
  char STRING_LIST[MXVAROBS_TEXT][20] ;   // string values per OBS row

  int NOBS_READ ;
  int NSPEC_READ ;
  int NLAM_READ ;

} TEXT_FILE_INFO ;


void WR_SNTEXTIO_DATAFILE(char *OUTFILE);
void wr_sntextio_datafile__(char *OUTFILE);

void wr_dataformat_text_HEADER(FILE *fp ) ;
void wr_dataformat_text_HOSTGAL(FILE *fp) ;
void wr_dataformat_text_SIMPAR(FILE *fp ) ;
void wr_dataformat_text_SNPHOT(FILE *fp ) ;
void wr_dataformat_text_SNSPEC(FILE *fp ) ;

void RD_SNTEXTIO_INIT(void); // one-time init
void rd_sntextio_init__(void);

int RD_SNTEXTIO_PREP(int MSKOPT, char *PATH, char *VERSION);
int rd_sntextio_prep__(int *MSKOPT, char *PATH, char *VERSION);

int  rd_sntextio_list(void);
void rd_sntextio_global(void);
void rd_sntextio_varlist_obs(int *iwd_file);
void rd_sntextio_varlist_spec(int *iwd_file);

void rd_sntextio_malloc_list(int OPT, int NFILE) ;
void rd_sntextio_malloc_spec(int ISPEC, int NBLAM);

void RD_SNTEXTIO_EVENT(int OPTMASK, int ifile);
void rd_sntextio_event__(int *OPTMASK, int *ifile);
bool parse_SNTEXTIO_HEAD(int *iwd);
bool parse_SNTEXTIO_OBS(int *iwd);
bool parse_SNTEXTIO_SPEC(int *iwd);

// xxxvoid check_plusminus_TEXT(int *iwd_file, float *PTR_ERR);
void parse_plusminus_sntextio(char *word, char *key, int *iwd_file, 
			      float *PTR_VAL, float *PTR_ERR) ;

void copy_keyword_nocolon(char *key_in, char *key_out) ;

