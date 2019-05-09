/**************

  Created Jun 2009 by R.Kessler

  Extract underlying SN parameter distribution
  using simulations & data as in 
   G D'Agostini, NIM A, 362, 487 (1995)

 Note that Appendix D of K09 corresponds to the
 above method with 1 iteration (i.e, P0_UNFOLD=1)

 Jun 2011: revived for the Smith analysis comparing
           distributions vs. host-galaxy type

   - abort if requested fitres variable is not found.
   - fix dumb bug filling 2-D plots (leftover from previous debugging?)
   - allow command-line override for a few variables:
      DATA_FILE  DATA_PATH  HFILE_OUT   N_ITERATION
   - write UNFOLDED_AVG and UNFOLDED_RMS for each parameter


  Feb 2013: update to compile with c++. Comment out wrhbook.

  May 7, 2013: (v10_28b)
   - wrhbook -> mkplots using sntools_output.c for either hbook or root.  

   - Input key HFILE_OUT replaced by PLOTFILE_OUT to accept either
     .his or .root extension. Legacy key HFILE_OUT still accepted.

   - hbook and hf1/hf2 calls replaced with SNHIST_INIT, SNHIST_FILL.
  
   - These updates have NOT been tested ... next user beware !

  Apr 26 2014:
    replace OPEN_HFILE & OPEN_ROOTFILE with generic wrapper
    TABLEFILE_OPEN().
    --> compiles & links, but not tested.

*************/
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "sntools.h"
#include "sntools_output.h"


// ##########################
//
// function declarations
//
// ##########################

void init_unfold(void);
void rd_input(void);
void unfold_input_override(void);
void rd_fitres(int itype, int itype2);
void fill_TABLE(int itype);  // store data in binned table
void fill_MIGRATION_TABLE(int IZACC, int IB1ACC, int IB2ACC,
			  int IZFIT, int IB1FIT, int IB2FIT, double WGTGEN ) ;

void  SET_TABLEBINS(char *string, int ipar);
void  SET_INDEXMAP(void);
void  GENBIN_LOOP(char *funopt);
void  UNFOLD_ZERO(int IZ, int IBIN1, int IBIN2 ) ;
void  UNFOLD_ADD(int IZ, int IBIN1, int IBIN2) ;  // evaluate Eq. D2 of K09
void  UNFOLD_RENORM(int IBIN1, int IBIN2 ) ;
void  PSIM_ADD(int IZ, int IBIN1, int IBIN2) ;  // sum PSIM_SUM array

int PARVAL2BIN(int ipar, double val);

void mkplots(void);
void PLOT_MIGRATION(int iplot) ;
void PLOT_DATAPRED(int ipar);
void prep_bindump(void);
void UNFOLDED_AVG_RMS(int ipar, int hid);

void  DMP_UNFOLD(int IZ, int IBIN1, int IBIN2, int imig );
void  DMP_ARRAYVAL(int itype, int ipar, int isn);
void  TEST_PARVAL2BIN(void);

//extern float hstati_(int *HID, int *ICASE, char *CHOICE, int *NUM, int len);


// ##########################
//
//      global variables.
//
// ##########################

#define MAXTYPE        8
#define ITYPE_DATA     1
#define ITYPE_DATAGEN  2 // sim-truth when data is from sim    
#define ITYPE_SIMFIT   3 // read from same file as SIMACC
#define ITYPE_SIMACC   4 // generated values after acceptances
#define ITYPE_SIMGEN   5 // generated values
#define ITYPE_UNFOLD   6 // unfolded values ; i.e, result


#define MAXPAR 3  // mandatory Z + two user parameters to unfold
#define IPAR_Z  0 // index for redshift
#define IPAR_1  1 // index for 1st param
#define IPAR_2  2 // index for 2nd param

#define MAXBINTOT 100000 // max for total number of bins
#define MAXBIN 104       // max number of bins to store per parameter
#define MAXZBIN 20       // fewere Z-bins to save memory
#define MAXSN  2000000   // max SN to store
#define MAXBINDUMP 50    // max bins to dump detailed info in UNFOLD()


#define OPT_UNFOLD_ZERO    1
#define OPT_PSIM_ADD       2
#define OPT_UNFOLD_ADD     3  // used in GENBIN_LOOP
#define OPT_UNFOLD_RENORM  4


struct INPUTS {
  char inFile[200];

  int N_ITERATION ; // number of unfolding iterations
  int NREAD_RANGE[MAXTYPE][2];

  char PATH[MAXTYPE][200];
  char FITRES_FILE[MAXTYPE][200];

  int  USETYPE[MAXTYPE];

  // not that IPAR=0 is for mandatory redshift
  char PARNAME[MAXPAR][MAXTYPE][40]; // parname(ipar=0,1,2: type : len )
 
  double  RANGE_PAR[MAXPAR][2]; // min,max
  double  BINSIZE_PAR[MAXPAR];  // binsize for each parameter
  int     NBIN_PAR[MAXPAR];     // Nbin for each parameter

  int NPAR; // number of parameters

  int NTYPE ;
  char TYPENAME[MAXTYPE][20];

  double MIGBIN_RADIUS;   // migration bin radius
  double MIGBIN_SQRADIUS; 

  char PLOTFILE_OUT[200];

  int NBINDUMP;
  double DUMPVAL[MAXBINDUMP][MAXPAR];
  int    IBINDUMP[MAXBINDUMP][MAXPAR]; // index refers to underlying 

  int    NMIGPLOT;  // number of requested migration bin plots
  double MIGVAL[MAXBINDUMP][MAXPAR];

  int DOMIGRATION_FLAG;  // default = 1

}  INPUTS ;


struct ARRAY {
  double PAR[MAXPAR][MAXSN];
  int N;
} ARRAY[MAXTYPE];


int NBINTOT ; // total number of bins to process
int INDEXMAP[MAXZBIN][MAXBIN][MAXBIN]; // translate multi-index to 1 index

struct INDEXMAP_INV {
  int IZ ;
  int I1 ;
  int I2 ;
} INDEXMAP_INV[MAXBINTOT];


struct TABLE {
  double N_OVERFLOW;
  double N_FILLED ;
  //  double ENTRIES[MAXZBIN][MAXBIN][MAXBIN]; // z, par1, par2
  double ENTRIES[MAXBINTOT];
} TABLE[MAXTYPE];


double TABLE_BINVALUES[MAXPAR][MAXBIN];  // bin-centered values like HBOOK

double P_UNFOLD[MAXBIN][MAXBIN] ;       // Eq D2 unfolding function
double P0_UNFOLD[MAXBIN][MAXBIN] ;      // initial guess of P_UNFOLD
double PSIM_SUM[MAXZBIN][MAXBIN][MAXBIN];  // used to normalize each Psim


double P_UNFOLD_RENORM;            // global renorm factor
double PSUM_UNFOLD;                // sum of P_UNFOLD


#define MAXMIGBIN 100
struct MIGRATION_TABLE {
  short int NMIGBIN ;               // number of bins stored
  float CONTAIN_FRAC ;              // contain fraction (ideal=100%)

  short int IBINZ_NEAR[MAXMIGBIN];  // underlying bin index
  short int IBIN1_NEAR[MAXMIGBIN];  // underlying bin index
  short int IBIN2_NEAR[MAXMIGBIN];  // idem
  double    NSIMFIT_NEAR[MAXMIGBIN];   // SIMACC entries in this bin
} MIGRATION_TABLE[MAXZBIN][MAXBIN][MAXBIN]; // fitted bins


// ================================
int main(int argc, char **argv) {

  int i, ITER ;
  double XN ;

  char fnam[20] = "main";

  // ------------ BEGIN ------------

  sprintf(BANNER,"Begin execution of snpar_unfold.exe  " );
  print_banner(BANNER);


  if ( argc >= 2 ) {
    sprintf(INPUTS.inFile, "%s", argv[1] );
    NARGV_LIST = argc ;
    for ( i = 0; i < NARGV_LIST ; i++ ) {
      sprintf( ARGV_LIST[i], "%s", argv[i] );
      USE_ARGV_LIST[i] = 0 ;
    }
    USE_ARGV_LIST[0] = 0 ;
    USE_ARGV_LIST[1] = 0 ;
  }
  else {
    sprintf(c1err,"Must give input filename as argument.");
    errmsg(SEV_FATAL, 0, fnam, c1err, ""); 
  }


  // read input file
  rd_input();

  // check for command-line overrides
  unfold_input_override();

  //  TEST_PARVAL2BIN();

  // init TABLES
  init_unfold();

  // set INDEXMAP 
  SET_INDEXMAP();

  // ------------------------------------------------
  // for each type, read fitres file & store table
  print_banner("Read FITRES Files & Store Tables");

  rd_fitres(ITYPE_DATA,-1);
  fill_TABLE(ITYPE_DATA);

  rd_fitres(ITYPE_DATAGEN,-1);
  fill_TABLE(ITYPE_DATAGEN);

  rd_fitres(ITYPE_SIMGEN,-1);
  fill_TABLE(ITYPE_SIMGEN);

  rd_fitres(ITYPE_SIMACC,ITYPE_SIMFIT);
  fill_TABLE(ITYPE_SIMFIT);
  fill_TABLE(ITYPE_SIMACC); 

  // --------------------------------
  //  prep_bindump();

  // -------------------
  // prepare loop over underlying bins

  ITER = 0;
  while ( ITER < INPUTS.N_ITERATION ) {

    ITER++ ;

    sprintf(BANNER," UNFOLD ITERATION %d : ", ITER);
    sprintf(BANNER,"%s Extract underlying %s vs. %s", BANNER
	   ,INPUTS.PARNAME[IPAR_1][ITYPE_SIMGEN]
	   ,INPUTS.PARNAME[IPAR_2][ITYPE_SIMGEN] );
    print_banner(BANNER) ;

    GENBIN_LOOP("UNFOLD_ZERO"); // set P_UNFOLD=0

    // sum PSIM over fitted bins
    GENBIN_LOOP("PSIM_ADD"); 

    // compute Eq. D2
    GENBIN_LOOP("UNFOLD_ADD");

    // renormalize P_UNFOLD 
    XN = TABLE[ITYPE_SIMGEN].N_FILLED;
    P_UNFOLD_RENORM  = XN / PSUM_UNFOLD ;
    //  printf("\t Renormalize P_UNFOLD -> *= %le \n", P_UNFOLD_RENORM  ); 

    GENBIN_LOOP("UNFOLD_RENORM");

  } // end of ITER loop


  mkplots();


} // end of main


// ==================================
void rd_input(void) {

  FILE *fp;
  char 
    c_get[200]
    ,string[200]
    ,*ptrFile
    ,*ptrname
    ,fnam[20] = "rd_input"
    ;

  int 
    NTYPE
    ,itype, itype2
    ,ipar
    ,NBIN
    ,i
    ,NTMP
    ;

  double xval[10], tmp ;

  // ---------- BEGIN -------
	 
  ptrFile = INPUTS.inFile;

  if ( (fp = fopen(ptrFile, "rt"))==NULL ) {       
    sprintf ( c1err, "Cannot open input file :" );
    sprintf ( c2err," '%s' ", ptrFile);
    errmsg(SEV_FATAL, 0, fnam, c1err, c2err); 
  }


  sprintf(BANNER, "Read user input file: %s", ptrFile );
  print_banner(BANNER);


  // set some defaults

  INPUTS.N_ITERATION = 1;
  INPUTS.DOMIGRATION_FLAG = 1 ;  // default = 1
  INPUTS.MIGBIN_RADIUS = 5.0 ;

  INPUTS.NPAR = 2; // fixed for now ... 

  NTYPE = INPUTS.NTYPE = 5 ; // excludes UNFOLDed
  sprintf( INPUTS.TYPENAME[ITYPE_DATA],     "DATA"    );
  sprintf( INPUTS.TYPENAME[ITYPE_DATAGEN],  "DATAGEN" );
  sprintf( INPUTS.TYPENAME[ITYPE_SIMFIT],   "SIMFIT"  );
  sprintf( INPUTS.TYPENAME[ITYPE_SIMACC],   "SIMACC"  );
  sprintf( INPUTS.TYPENAME[ITYPE_SIMGEN],   "SIMGEN"  );
  sprintf( INPUTS.TYPENAME[ITYPE_UNFOLD],   "UNFOLD"  );

  INPUTS.NBINDUMP = 0 ;

  for ( itype=0 ; itype < MAXTYPE; itype++ ) {
    INPUTS.NREAD_RANGE[itype][0] = 0 ;
    INPUTS.NREAD_RANGE[itype][1] = 9999999 ;
    INPUTS.USETYPE[itype] = 0;
  }


  while( (fscanf(fp, "%s", c_get)) != EOF) {

    if ( strcmp(c_get,"MIGRATION_BIN_RADIUS:") == 0 ) {
      readdouble(fp, 1, &INPUTS.MIGBIN_RADIUS );
      INPUTS.MIGBIN_SQRADIUS = 
	INPUTS.MIGBIN_RADIUS * INPUTS.MIGBIN_RADIUS ;

      if ( INPUTS.MIGBIN_RADIUS <= 0.0 ) INPUTS.DOMIGRATION_FLAG=0;
    }

    if ( strcmp(c_get,"PLOTFILE_OUT:") == 0 ) 
      {  readchar(fp, INPUTS.PLOTFILE_OUT); }
    if ( strcmp(c_get,"HFILE_OUT:") == 0 )   // allow legacy key
      {  readchar(fp, INPUTS.PLOTFILE_OUT); }


    if ( strcmp(c_get,"N_ITERATION:") == 0 ) 
      readint(fp, 1, &INPUTS.N_ITERATION );


    if ( strcmp(c_get,"BINDUMP:") == 0 ) {
      INPUTS.NBINDUMP++ ;
      NTMP = INPUTS.NBINDUMP ;

      for ( ipar=0; ipar <= 2; ipar++ ) {
	readdouble(fp, 1, &tmp );
	INPUTS.DUMPVAL[NTMP][ipar] = tmp ;
	INPUTS.IBINDUMP[NTMP][ipar] =  PARVAL2BIN(ipar,tmp);
      }
    }


    if ( strcmp(c_get,"MIGRATION_PLOT:") == 0 ) {
      INPUTS.NMIGPLOT++ ;
      NTMP = INPUTS.NMIGPLOT ;
      readdouble(fp, 1, &INPUTS.MIGVAL[NTMP][IPAR_Z] );
      readdouble(fp, 1, &INPUTS.MIGVAL[NTMP][IPAR_1] );
      readdouble(fp, 1, &INPUTS.MIGVAL[NTMP][IPAR_2] );
    }

    for ( itype=1; itype <= NTYPE; itype++ ) {

      if ( itype == ITYPE_SIMFIT ) continue ;

      if ( itype == ITYPE_SIMACC ) 
	itype2 = ITYPE_SIMFIT ;  // read this from same file
      else
	itype2 = -1;

      ptrname = INPUTS.TYPENAME[itype] ;

      sprintf(string,"%s_PATH:",  ptrname );
      if ( strcmp(c_get,string) == 0 ) 
	readchar(fp, INPUTS.PATH[itype] );

      sprintf(string,"%s_FILE:",  ptrname );
      if ( strcmp(c_get,string) == 0 ) {
	readchar(fp, INPUTS.FITRES_FILE[itype] );
	INPUTS.USETYPE[itype] = 1 ;

	if ( itype == ITYPE_SIMACC )
	  INPUTS.USETYPE[ITYPE_SIMFIT] = 1 ; 
      }

      sprintf(string,"%s_PARNAMEZ:",  ptrname );
      if ( strcmp(c_get,string) == 0 ) {
	readchar(fp, INPUTS.PARNAME[IPAR_Z][itype] );
	if ( itype2 > 0 ) 
	  { readchar(fp, INPUTS.PARNAME[IPAR_Z][itype2] ); }
      }

      sprintf(string,"%s_PARNAME1:",  ptrname );
      if ( strcmp(c_get,string) == 0 ) {
	readchar(fp, INPUTS.PARNAME[IPAR_1][itype] );
	if ( itype2 > 0 ) 
	  readchar(fp, INPUTS.PARNAME[IPAR_1][itype2] );
      }

      sprintf(string,"%s_PARNAME2:",  ptrname );
      if ( strcmp(c_get,string) == 0 ) {
	readchar(fp, INPUTS.PARNAME[IPAR_2][itype] );
	if ( itype2 > 0 ) 
	  readchar(fp, INPUTS.PARNAME[IPAR_2][itype2] );
      }

      sprintf(string,"%s_NREAD:",  ptrname );
      if ( strcmp(c_get,string) == 0 ) {
	readint(fp, 2, INPUTS.NREAD_RANGE[itype] );
      }


    } // end of itype loop

    for ( ipar = 0; ipar <= INPUTS.NPAR ; ipar++ ) {

      if ( ipar == 0 ) 
	sprintf(string,"BINDEF_PARZ:" ) ;
      else
	sprintf(string,"BINDEF_PAR%d:", ipar ) ;

      if ( strcmp(c_get,string) == 0 ) {
	readdouble( fp, 3, xval  );
	INPUTS.RANGE_PAR[ipar][0] = xval[0] ;
	INPUTS.RANGE_PAR[ipar][1] = xval[1] ;
	INPUTS.BINSIZE_PAR[ipar]  = xval[2] ;

	SET_TABLEBINS(string,ipar);
      } 
    }


  } // end of fscanf loop

  fclose(fp);

  // summarize user input

  for( itype=1; itype <= NTYPE; itype++ ) {
    ptrname = INPUTS.TYPENAME[itype] ;
    printf(" -------- \n");
    printf(" %s : \n", ptrname );

    if ( itype == ITYPE_SIMFIT ) {
      sprintf(INPUTS.PATH[itype],"Same as for SIMACC");
      sprintf(INPUTS.FITRES_FILE[itype],"Same as for SIMACC");
    }

    if ( itype == ITYPE_DATAGEN ) {
      sprintf(INPUTS.PATH[itype],"%s", INPUTS.PATH[ITYPE_DATA] ) ;
    }

    printf("   PATH: %s \n", INPUTS.PATH[itype] );
    printf("   FILE: %s \n", INPUTS.FITRES_FILE[itype] );

    printf("   FITRES PARNAMES(z,1,2): %s %s %s \n"
	   ,INPUTS.PARNAME[IPAR_Z][itype] 
	   ,INPUTS.PARNAME[IPAR_1][itype] 
	   ,INPUTS.PARNAME[IPAR_2][itype] 
	   );
  }


  printf(" -------- \n");
  printf("  MIGRATION-BIN STORAGE RADIUS: %5.1f \n", INPUTS.MIGBIN_RADIUS);
  printf("  Number of MIGRATION-BIN plots: %d \n", INPUTS.NMIGPLOT );

  printf("\n Done reading input file. \n\n");

} // end of rd_input


// ====================================
void  unfold_input_override(void) {

  int i, ilast, iuse;
  char *cptr ;
  char fnam[40] = "unfold_input_override" ;

  
  // ----------- BEGIN ------------

  i = ilast = 2;

  while ( i < NARGV_LIST ) {
    printf("  PROCESS COMMAND LINE ARG: %s \n", ARGV_LIST[i] );


    if ( strcmp( ARGV_LIST[i], "DATA_FILE" ) == 0 ) {
      cptr = INPUTS.FITRES_FILE[ITYPE_DATA] ;
      i++ ; sscanf(ARGV_LIST[i] , "%s", cptr ); 
    }

    if ( strcmp( ARGV_LIST[i], "DATA_PATH" ) == 0 ) {
      cptr = INPUTS.PATH[ITYPE_DATA] ;
      i++ ; sscanf(ARGV_LIST[i] , "%s", cptr ); 
    }

    if ( strcmp( ARGV_LIST[i], "HFILE_OUT" ) == 0 ) {
      i++ ; sscanf(ARGV_LIST[i] , "%s", INPUTS.PLOTFILE_OUT);
    }
    if ( strcmp( ARGV_LIST[i], "PLOTFILE_OUT" ) == 0 ) {
      i++ ; sscanf(ARGV_LIST[i] , "%s", INPUTS.PLOTFILE_OUT);
    }

    if ( strcmp( ARGV_LIST[i], "N_ITERATION" ) == 0 ) {
      i++ ; sscanf(ARGV_LIST[i] , "%d", &INPUTS.N_ITERATION );
    }


    // -------------------
    if ( i > ilast ) {
      for ( iuse = ilast; iuse <= i; iuse++ ) 
	USE_ARGV_LIST[iuse] = 1;
    }
    i++ ;  ilast=i;

  } // end loop over i

} // end of unfold_input_override


// =============================
void init_unfold(void) {
  // init binned table array

  int iz, i1, i2, itype, j ;

  // ---------- BEGIN -----------

  PSUM_UNFOLD = 0.0 ;

  for ( itype=0; itype < MAXTYPE; itype++ ) {

    TABLE[itype].N_OVERFLOW = 0. ;
    TABLE[itype].N_FILLED   = 0. ;

    for ( j=0; j < MAXBINTOT; j++ ) 
      TABLE[itype].ENTRIES[j] = 0.0 ;

  }

  for ( iz=0; iz < MAXZBIN; iz++ ) {
    for ( i1=0; i1 < MAXBIN; i1++ ) {
      for ( i2=0; i2 < MAXBIN; i2++ ) {

	if ( iz == 0 ) {
	  P_UNFOLD[i1][i2]     = 0.0 ;
	  P0_UNFOLD[i1][i2]    = 1.0 ;  
	}
 
	PSIM_SUM[iz][i1][i2] = 0.0 ;
	MIGRATION_TABLE[iz][i1][i2].NMIGBIN      = 0 ;
	MIGRATION_TABLE[iz][i1][i2].CONTAIN_FRAC = 0.0 ;

	for ( j=0; j<MAXMIGBIN; j++ ) {
	  MIGRATION_TABLE[iz][i1][i2].NSIMFIT_NEAR[j] = 0.0 ;
	}

      }
    }
  }   // end iz loop

}  // end of init_unfold


// ============================
void SET_TABLEBINS(char *string, int ipar) {

  // error checking and set some global TABLEBIN arrays

  double xmin,  xmax, xbin, tmp, xi;
  char fnam[20] = "SET_TABLEBINS" ;
  int NBIN, NBIN_MAX, ibin ;

  // --------- BEGIN ---------

  xmin = INPUTS.RANGE_PAR[ipar][0] ;
  xmax = INPUTS.RANGE_PAR[ipar][1] ;
  xbin = INPUTS.BINSIZE_PAR[ipar] ;

  if ( ipar == IPAR_Z ) 
    NBIN_MAX = MAXZBIN ;
  else
    NBIN_MAX = MAXBIN ;

  if ( xbin <= 0.0 ) {
    sprintf(c1err,"Invalid binsize=%f for %s", xbin, string);
    sprintf(c2err,"Check user input file. ");
    errmsg(SEV_FATAL, 0, fnam, c1err, c2err); 
  }

  tmp = (xmax - xmin)/ xbin ;
  NBIN = INPUTS.NBIN_PAR[ipar] = (int)(tmp+0.000001);
  if ( NBIN >= NBIN_MAX ) {
    sprintf(c1err,"NBIN=%d exceeds array bound for %s", NBIN, string);
    sprintf(c2err,"Check user input file. ");
    errmsg(SEV_FATAL, 0, fnam, c1err, c2err); 
  }

  // set TABLE_BINVALUES

  for ( ibin=1; ibin <= NBIN; ibin++ ) {
    xi = (double)ibin ;
    tmp = xmin + xbin * ( xi - 0.5 );
    TABLE_BINVALUES[ipar][ibin] = tmp ;

  }

  printf("\t %4d %8s  TABLE BINS from  %7.3f  to  %7.3f \n",
	 NBIN, INPUTS.PARNAME[ipar][ITYPE_SIMGEN], xmin, xmax ) ;


} // end of SET_TABLEBINS


// ===========================
void SET_INDEXMAP(void) {

  int NBZ, NB1, NB2; 
  int IZ,  IB1, IB2, INDEX ;

  char fnam[20] = "SET_INDEXMAP";

  // ----------- BEGIN ------------

  print_banner("Set INDEXMAP pointers");

  NBZ = INPUTS.NBIN_PAR[IPAR_Z];
  NB1 = INPUTS.NBIN_PAR[IPAR_1];
  NB2 = INPUTS.NBIN_PAR[IPAR_2];

  INDEX = 0;

  for ( IZ=0; IZ <= NBZ; IZ++ ) {  // iz=0 => all z-bins
    for ( IB1=1; IB1 <= NB1; IB1++ ) {
      for ( IB2=1; IB2 <= NB2; IB2++ ) {           

	INDEX++ ;
	if ( INDEX >= MAXBINTOT ) {
	  sprintf(c1err,"INDEXMAP = %d exceeds array bound.", INDEX);
	  sprintf(c2err,"Check parameter MAXBINTOT");
	  errmsg(SEV_FATAL, 0, fnam, c1err, c2err ); 
	}

	INDEXMAP[IZ][IB1][IB2] = INDEX;

	INDEXMAP_INV[INDEX].IZ = IZ ;
	INDEXMAP_INV[INDEX].I1 = IB1 ;
	INDEXMAP_INV[INDEX].I2 = IB2 ;
      }
    }
  }

  NBINTOT = INDEX ;
  printf("\t Finished filling INDEXMAP for %d indices\n", NBINTOT );


} // end of SET_INDEXMAP

// ===========================
void rd_fitres(int itype, int itype2 ) {

  // Read itype variables from itype file.
  // If "itype2" > 0, then read these variables
  // from same file.
  // Note that ARRAY starts at isn=1 (not zero)

  char 
    fnam[20] = "rd_fitres"
    ,local_filename[200]
    ,*ptrFile, *parName
    ;

  int NVAR, index, NSN, ipar, N0, N1 ;

  // -------- BEGIN ---------

  if ( INPUTS.USETYPE[itype] == 0 ) return ;

  ptrFile = local_filename ;
  sprintf(local_filename, "%s/%s", 
	  INPUTS.PATH[itype], INPUTS.FITRES_FILE[itype] );

  printf("\n");

  TABLEFILE_INIT();
  int IFILETYPE = TABLEFILE_OPEN(ptrFile,"read");
  NVAR = SNTABLE_READPREP(IFILETYPE,"FITRES");
  //  NVAR = fitresFile_init(ptrFile);

  for ( ipar=0; ipar <= INPUTS.NPAR; ipar++ ) {
    parName = INPUTS.PARNAME[ipar][itype] ;
    index   = SNTABLE_READPREP_VARDEF(parName,&ARRAY[itype].PAR[ipar][1],MAXSN,1) ;

    if ( index < 0 ) {
      sprintf(c1err,"Could not find FITRES variable '%s' in", parName);
      sprintf(c2err,"%s", local_filename);
      errmsg(SEV_FATAL, 0, fnam, c1err, c2err); 
    }

    // check option to read more variables from same file.
    if ( itype2 > 0 ) {
      parName = INPUTS.PARNAME[ipar][itype2] ;
      index = SNTABLE_READPREP_VARDEF(parName, &ARRAY[itype2].PAR[ipar][1], MAXSN, 1 ) ;
      if ( index < 0 ) {
	sprintf(c1err,"Could not find FITRES variable '%s' in", parName);
	sprintf(c2err,"%s", local_filename);
	errmsg(SEV_FATAL, 0, fnam, c1err, c2err); 
      }
    }

  }


  // read file

  NSN = SNTABLE_READ_EXEC();
  ARRAY[itype].N = NSN ;

  // print extra message if fewer SN will be used.

  N0 = INPUTS.NREAD_RANGE[itype][0] ;
  N1 = INPUTS.NREAD_RANGE[itype][1] ; 
  if ( N1 > NSN ) N1 = NSN ;

  if ( N0 > 1 || N1 < NSN ) 
    printf("\t ==> But will only use entries %d to %d \n", N0, N1 );


  if ( itype2 > 0 )  ARRAY[itype2].N = NSN ;


  if ( NSN >= MAXSN ) {
    sprintf(c1err,"NNS=%d exceeds array bound of %d", NSN, MAXSN);
    sprintf(c2err,"Check fitres file, or increase MAXSN ");
    errmsg(SEV_FATAL, 0, fnam, c1err, c2err); 
  }


} // end of rd_fitres


//===========================================
void fill_TABLE(int itype) {

  // Jun 2009: fill TABLE array for "itype"

  int 
    isn
    ,ipar
    ,iz,  i1,  i2 
    ,iz2, i12, i22 
    ,ibin[10]
    ,ibin2[10]
    ,INDEX, INDEX0
    ,ISVALID
    ,LMIGBIN
    ;

  double 
    tmp
    ,parval[10]
    ,parval2[10]
    ,WGTGEN
    ;

  // ---------- BEGIN ---------

  if ( INPUTS.USETYPE[itype] == 0 ) return ;


  LMIGBIN = 0;
  if ( itype == ITYPE_SIMACC ) { LMIGBIN = 1; }

  for( isn=1; isn <= ARRAY[itype].N; isn++ ) {

    if ( isn < INPUTS.NREAD_RANGE[itype][0] ) continue ;
    if ( isn > INPUTS.NREAD_RANGE[itype][1] ) continue ;

    ISVALID = 1;

    for ( ipar=0; ipar <= INPUTS.NPAR; ipar++ ) {


      parval[ipar]  = ARRAY[itype].PAR[ipar][isn] ;
      ibin[ipar]    = PARVAL2BIN( ipar, parval[ipar] );
      if ( ibin[ipar] < 1 )                     ISVALID = 0;
      if ( ibin[ipar] > INPUTS.NBIN_PAR[ipar] ) ISVALID = 0;

      if ( LMIGBIN ) {
	parval2[ipar] = ARRAY[ITYPE_SIMFIT].PAR[ipar][isn] ;
	ibin2[ipar]   = PARVAL2BIN( ipar, parval2[ipar] );
      }

    } // end of ipar loop

    iz = ibin[IPAR_Z]; i1 = ibin[IPAR_1];  i2 = ibin[IPAR_2] ;

    INDEX  = INDEXMAP[iz][i1][i2];
    INDEX0 = INDEXMAP[0][i1][i2];

    if ( ISVALID == 1 ) {

      WGTGEN = 1.0;

      TABLE[itype].ENTRIES[INDEX]  += WGTGEN ;
      TABLE[itype].ENTRIES[INDEX0] += WGTGEN ;
      TABLE[itype].N_FILLED        += WGTGEN ;

      if ( LMIGBIN ) {
	iz2 = ibin2[IPAR_Z]; 
	i12 = ibin2[IPAR_1];  
	i22 = ibin2[IPAR_2] ;
	fill_MIGRATION_TABLE( iz,i1,i2, iz2,i12,i22, WGTGEN );
      }
    }
    else
      TABLE[itype].N_OVERFLOW += WGTGEN ;

  } // end if 'i' loop over ARRAY


  printf("  fill_TABLE(): filled %d table entries (%d overflow)\n",
	 (int)TABLE[itype].N_FILLED, (int)TABLE[itype].N_OVERFLOW );

  fflush(stdout);

} // end of fill_TABLE


// ======================================
int PARVAL2BIN(int ipar, double val) {

  double tmp, valmin, valmax ;
  int ibin;

  // ------- BEGIN ---------

  ibin = -9;
  valmin = INPUTS.RANGE_PAR[ipar][0] ;
  valmax = INPUTS.RANGE_PAR[ipar][1] ;

  if ( val < valmin ) return ibin ;
  if ( val > valmax ) return ibin ;

  tmp = ( val - valmin + 0.000001 ) ;
  tmp /= INPUTS.BINSIZE_PAR[ipar] ;
  ibin = (int)tmp + 1 ; // bins start at 1

  return ibin;

} // end of VAL2BIN


// ===================================
void fill_MIGRATION_TABLE(int IZACC, int IB1ACC, int IB2ACC,
			  int IZFIT, int IB1FIT, int IB2FIT, 
			  double WGTGEN ) {

  // Jun 2009
  // Fill migration table for this SN.
  // Inputs are underlayin (acc) bins and fitted (fit) bins

  double 
    RSQMAX, RSQDIF, DIFZ, DIF1, DIF2
    ,x1, x2 
    ;

  int 
    LDMP
    ,NMIGBIN, IFIT_USED
    ,ifit, izfit, ib1fit, ib2fit
    ,INDEX
    ;

  char fnam[40] = "fill_MIGRATION_TABLE";

  // -------------- BEGIN --------------

  if ( INPUTS.DOMIGRATION_FLAG == 0 ) return ;

  // make sure indices are well defined
  if ( IZACC  < 1 ) goto SKIPPY ;
  if ( IB1ACC < 1 ) goto SKIPPY ;
  if ( IB2ACC < 1 ) goto SKIPPY ;

  if ( IZFIT  < 1 ) goto SKIPPY ;
  if ( IB1FIT < 1 ) goto SKIPPY ;
  if ( IB2FIT < 1 ) goto SKIPPY ;

  if ( IZACC  > INPUTS.NBIN_PAR[IPAR_Z] ) goto SKIPPY ;
  if ( IB1ACC > INPUTS.NBIN_PAR[IPAR_1] ) goto SKIPPY ;
  if ( IB2ACC > INPUTS.NBIN_PAR[IPAR_2] ) goto SKIPPY ;

  if ( IZFIT  > INPUTS.NBIN_PAR[IPAR_Z] ) goto SKIPPY ;
  if ( IB1FIT > INPUTS.NBIN_PAR[IPAR_1] ) goto SKIPPY ;
  if ( IB2FIT > INPUTS.NBIN_PAR[IPAR_2] ) goto SKIPPY ;

  RSQMAX = INPUTS.MIGBIN_SQRADIUS ;

  DIFZ = (double)(IZACC  - IZFIT ) - 0.5 ;
  DIF1 = (double)(IB1ACC - IB1FIT) - 0.5 ;
  DIF2 = (double)(IB2ACC - IB2FIT) - 0.5 ;
  RSQDIF = DIFZ*DIFZ + DIF1*DIF1 + DIF2*DIF2 ;


  // bail if fitted bin value is too far from underlying value.
  if ( RSQDIF > RSQMAX ) goto SKIPPY ;

  // check debug-dump flag
  LDMP = 0;
  if ( IZACC == -2 && IB1ACC == 7 && IB2ACC == 10 ) {
    LDMP = 1 ;
    printf(" %s DUMP for %s=%5.3f  %s=%6.3f  %s=%6.3f \n"
	   ,fnam
	   ,INPUTS.PARNAME[IPAR_Z][ITYPE_SIMFIT]
	   ,TABLE_BINVALUES[IPAR_Z][IZACC] 
	   ,INPUTS.PARNAME[IPAR_1][ITYPE_SIMFIT]
	   ,TABLE_BINVALUES[IPAR_1][IB1ACC] 
	   ,INPUTS.PARNAME[IPAR_2][ITYPE_SIMFIT]
	   ,TABLE_BINVALUES[IPAR_2][IB2ACC] 
	   );
  }



  NMIGBIN = MIGRATION_TABLE[IZACC][IB1ACC][IB2ACC].NMIGBIN;

  // check if this FIT-bin is already stored;
  IFIT_USED = 0;
  for ( ifit=1; ifit <= NMIGBIN; ifit++ ) {
    izfit  = MIGRATION_TABLE[IZACC][IB1ACC][IB2ACC].IBINZ_NEAR[ifit] ;
    ib1fit = MIGRATION_TABLE[IZACC][IB1ACC][IB2ACC].IBIN1_NEAR[ifit] ;
    ib2fit = MIGRATION_TABLE[IZACC][IB1ACC][IB2ACC].IBIN2_NEAR[ifit] ;
    if (  izfit == IZFIT && ib1fit == IB1FIT && ib2fit == IB2FIT ) 
      { IFIT_USED = ifit ; }
  }


  if ( IFIT_USED == 0 ) {
    NMIGBIN++ ;
    IFIT_USED = NMIGBIN ;
    MIGRATION_TABLE[IZACC][IB1ACC][IB2ACC].NMIGBIN = NMIGBIN ;
    MIGRATION_TABLE[IZACC][IB1ACC][IB2ACC].IBINZ_NEAR[NMIGBIN] = IZFIT ;
    MIGRATION_TABLE[IZACC][IB1ACC][IB2ACC].IBIN1_NEAR[NMIGBIN] = IB1FIT ;
    MIGRATION_TABLE[IZACC][IB1ACC][IB2ACC].IBIN2_NEAR[NMIGBIN] = IB2FIT ;

    if ( LDMP ) printf("\t NMIGBIN=%d for IB[1,2]FIT=%d,%d \n",
		       NMIGBIN, IB1FIT,IB2FIT );

    if ( NMIGBIN >= MAXMIGBIN ) {
      sprintf(c1err,"NMIGBIN=%d exceeds array bound for", NMIGBIN);
      sprintf(c2err,"IZ,I1,I2(ACC)=%d %d %d   IZ,I1,I2(FIT)=%d %d %d ",
	      IZACC, IB1ACC, IB2ACC,  IZFIT, IB1FIT, IB2FIT );
      errmsg(SEV_FATAL, 0, fnam, c1err, c2err); 
    }
  }



  // increment number of entries for this ACC+FIT
  MIGRATION_TABLE[IZACC][IB1ACC][IB2ACC].NSIMFIT_NEAR[IFIT_USED] += WGTGEN ;
  MIGRATION_TABLE[IZACC][IB1ACC][IB2ACC].NSIMFIT_NEAR[0]  += WGTGEN ;

  // sum over z-bins with iz=0 bin
  MIGRATION_TABLE[0][IB1ACC][IB2ACC].NSIMFIT_NEAR[IFIT_USED] += WGTGEN ;
  MIGRATION_TABLE[0][IB1ACC][IB2ACC].NSIMFIT_NEAR[0] += WGTGEN ;


  // update containment fraction
  INDEX = INDEXMAP[IZACC][IB1ACC][IB2ACC] ;
  x1 = MIGRATION_TABLE[IZACC][IB1ACC][IB2ACC].NSIMFIT_NEAR[0];
  x2 = TABLE[ITYPE_SIMACC].ENTRIES[INDEX];
  if ( x2 > 0.0 ) 
    MIGRATION_TABLE[IZACC][IB1ACC][IB2ACC].CONTAIN_FRAC = x1/x2 ;


  /** mark for deletion
  x1 = MIGRATION_TABLE[0][IB1ACC][IB2ACC].NSIMFIT_NEAR[0] ;
  x2 = TABLE[ITYPE_SIMACC].ENTRIES[0][IB1ACC][IB2ACC] ;
  if ( x2 > 0.0 ) 
    MIGRATION_TABLE[0][IB1ACC][IB2ACC].CONTAIN_FRAC = x1/x2 ;
  **/

 SKIPPY:
  x1 = 0.0;  // dummy 

} // end of fill_MIGRATION_TABLE


// ======================================
void GENBIN_LOOP(char *funopt) {

  // loop over generation bins and call function based on *funopt

  int IZ, IBIN1, IBIN2;
  int NBZ, NB1, NB2, OPT ;

  char fnam[20] = "GENBIN_LOOP" ;

  // --------- begin ----------

  NBZ = INPUTS.NBIN_PAR[IPAR_Z];
  NB1 = INPUTS.NBIN_PAR[IPAR_1];
  NB2 = INPUTS.NBIN_PAR[IPAR_2];

  // convert string-optoin into integer option

  if ( strcmp(funopt,"UNFOLD_ZERO") == 0 ) {
    OPT = OPT_UNFOLD_ZERO ;
  }
  else if ( strcmp(funopt,"PSIM_ADD") == 0 ) 
    OPT = OPT_PSIM_ADD ;

  else if ( strcmp(funopt,"UNFOLD_ADD") == 0 ) 
    OPT = OPT_UNFOLD_ADD ;

  else if ( strcmp(funopt,"UNFOLD_RENORM") == 0 ) {
    OPT = OPT_UNFOLD_RENORM ;
    NBZ = 1;
  }
  else {
    sprintf(c1err,"invalid funopt = '%s'", funopt );
    errmsg(SEV_FATAL, 0, fnam, c1err, "" ); 
  }


  printf("\t %s(%s) => OPT = %d \n", fnam, funopt, OPT );

  // -----------

  for ( IZ=1; IZ <= NBZ; IZ++ ) {
    for ( IBIN1=1; IBIN1 <= NB1; IBIN1++ ) {
      for ( IBIN2=1; IBIN2 <= NB2; IBIN2++ ) {           

	if ( OPT == OPT_UNFOLD_ZERO ) 
	  UNFOLD_ZERO(IZ,IBIN1,IBIN2) ;

	else if ( OPT == OPT_PSIM_ADD )
	  PSIM_ADD(IZ,IBIN1,IBIN2) ;

	else if ( OPT == OPT_UNFOLD_ADD )
	  UNFOLD_ADD(IZ,IBIN1,IBIN2) ;

	else if ( OPT == OPT_UNFOLD_RENORM ) 
	  UNFOLD_RENORM(IBIN1,IBIN2);

	
      }  // IBIN2
    }   // IBIN1
  }    // IZ 

} // end of LOOPSHELL

// =======================================
void PSIM_ADD (int IZ, int IBIN1, int IBIN2) {

  // Fill PSIM_SUM to use in unfolding equation.
  // IBIN1, IBIN2 refer to generated values
  // ibin1, ibin2 refer to measured (fitted) values

  int iz, ibin1, ibin2;
  int NMIGBIN, imig;

  double PSIM, XFIT, XGEN, XACC, P0 ;
  int i, INDEX, IFLAG_DUMP ;

  // --------------- BEGIN ----------------

  // bail if there are no generated events here

  INDEX = INDEXMAP[IZ][IBIN1][IBIN2] ;
  XGEN = TABLE[ITYPE_SIMGEN].ENTRIES[INDEX] ;
  XACC = TABLE[ITYPE_SIMACC].ENTRIES[INDEX] ;
  if ( XGEN <= 0.  ) return ;
  if ( XACC <= 0.  ) return ;

  P0 = P0_UNFOLD[IBIN1][IBIN2] ; // initial/last guess

  iz = IZ;

  // loop over migration bins 
  NMIGBIN = MIGRATION_TABLE[IZ][IBIN1][IBIN2].NMIGBIN;

  for ( imig=1; imig <= NMIGBIN ; imig++ ) {

    // ibin[1,2] are fitted bins
    iz      = MIGRATION_TABLE[IZ][IBIN1][IBIN2].IBINZ_NEAR[imig] ;
    ibin1   = MIGRATION_TABLE[IZ][IBIN1][IBIN2].IBIN1_NEAR[imig] ;
    ibin2   = MIGRATION_TABLE[IZ][IBIN1][IBIN2].IBIN2_NEAR[imig] ;

    XFIT = MIGRATION_TABLE[IZ][IBIN1][IBIN2].NSIMFIT_NEAR[imig];
    PSIM = XFIT / XGEN ;      

    // note that indices are fitted (not generated)
    PSIM_SUM[iz][ibin1][ibin2] += (PSIM*P0) ;
    
  } // end of imig loop


} // end of PSIM_ADD


// ==================================
void UNFOLD_ZERO(int IZ, int IBIN1, int IBIN2 ) {

  // zero out arrays before starting another iteration

  PSIM_SUM[IZ][IBIN1][IBIN2] = 0.0 ;

  if ( IZ == 1 ) {
    P_UNFOLD[IBIN1][IBIN2]  = 0.0 ;
    PSUM_UNFOLD             = 0.0 ;
  }

} // end of UNFOLD_ZERO


// ==================================
void UNFOLD_RENORM(int IBIN1, int IBIN2 ) {

  // Apply global re-norm to P_UNFOLD,
  // and set P0_UNFOLD for next iteration

  double XGEN ;
  // ----------- BEGIN -----------

  P_UNFOLD[IBIN1][IBIN2] *= P_UNFOLD_RENORM ;

  P0_UNFOLD[IBIN1][IBIN2] = P_UNFOLD[IBIN1][IBIN2] ;

} // end of UNFOLD_RENORM

// =======================================
void UNFOLD_ADD (int IZ, int IBIN1, int IBIN2) {

  // IBIN1, IBIN2 refer to generated values
  // ibin1, ibin2 refer to measured (fitted) values

  int iz, ibin1, ibin2, INDEX, index;
  int NMIGBIN, imig;

  double 
    PSIM, P0
    ,PSIM_WGT
    ,EFFSIM
    ,XDATA, XGEN, XFIT, XACC, PROB_MIG
    ,PRODUCT
    ;

  int i, IFLAG_DUMP ;

  char fnam[20] = "UNFOLD_ADD";

  // --------------- BEGIN ----------------


  // bail if there are no generated events here

  INDEX = INDEXMAP[IZ][IBIN1][IBIN2] ;
  XACC = TABLE[ITYPE_SIMACC].ENTRIES[INDEX] ;
  XGEN = TABLE[ITYPE_SIMGEN].ENTRIES[INDEX] ;
  if ( XGEN <= 0.0  ) return ;
  if ( XACC <= 0.0  ) return ;

  EFFSIM = XACC / XGEN ; 
  P0 = P0_UNFOLD[IBIN1][IBIN2] ; // initial/last guess

  // check for dump flag
  IFLAG_DUMP = 0;
  for ( i=1; i <= INPUTS.NBINDUMP; i++ ) {
    if ( IBIN1 == INPUTS.IBINDUMP[i][IPAR_1] &&
	 IBIN2 == INPUTS.IBINDUMP[i][IPAR_2] &&
	 IZ    == INPUTS.IBINDUMP[i][IPAR_Z] 
	 ) {
      IFLAG_DUMP = 1;
    }
  }


  if ( IFLAG_DUMP  ) DMP_UNFOLD(IZ, IBIN1, IBIN2, 0 );
  

  // loop over migration bins 
  NMIGBIN = MIGRATION_TABLE[IZ][IBIN1][IBIN2].NMIGBIN;

  for ( imig=1; imig <= NMIGBIN ; imig++ ) {

    // ibin[1,2] are fitted bins
    iz      = MIGRATION_TABLE[IZ][IBIN1][IBIN2].IBINZ_NEAR[imig] ;
    ibin1   = MIGRATION_TABLE[IZ][IBIN1][IBIN2].IBIN1_NEAR[imig] ;
    ibin2   = MIGRATION_TABLE[IZ][IBIN1][IBIN2].IBIN2_NEAR[imig] ;

    index = INDEXMAP[iz][ibin1][ibin2] ; 
    XDATA   = TABLE[ITYPE_DATA].ENTRIES[index] ;
    if ( XDATA <= 0.0 ) continue ;

    if ( IFLAG_DUMP )  DMP_UNFOLD(IZ, IBIN1, IBIN2, imig);

    XFIT = MIGRATION_TABLE[iz][IBIN1][IBIN2].NSIMFIT_NEAR[imig];
    PSIM = XFIT / XGEN ;      

    if ( INPUTS.DOMIGRATION_FLAG  )
      PSIM_WGT = PSIM_SUM[iz][ibin1][ibin2];
    else
      PSIM_WGT = 1.0 ;

    if ( PSIM_WGT <= 0.0 ) {
      sprintf(c1err,"Invalid PSIM_WGT=%f at iz=%d ibin1=%d ibin2=%d",
	      PSIM_WGT, iz,ibin1, ibin2);
      sprintf(c2err,"IBIN1=%d  IBIN2=%d", IBIN1, IBIN2 );
      errmsg(SEV_FATAL, 0, fnam, c1err, c2err); 
    }

    
    // See Eq. 3 of D'Agonstini NIM A 362, 487 (1995)
    PROB_MIG = (PSIM * P0 / PSIM_WGT) ;  // P(C_i | E_j)

    // Eq. 4 of above
    PRODUCT = XDATA * PROB_MIG / EFFSIM  ;

    P_UNFOLD[IBIN1][IBIN2] += PRODUCT ;
    PSUM_UNFOLD += PRODUCT ;

  } // end of imig loop

  if ( INPUTS.DOMIGRATION_FLAG == 0 ) {
    XDATA = TABLE[ITYPE_DATA].ENTRIES[INDEX] ;
    P_UNFOLD[ibin1][ibin2] += XDATA / EFFSIM ;
  }


} // end of UNFOLD_ADD



// ==========================
void mkplots(void) {

  // May 7 2013: re-written to use sntools_output.

  char *plotFile;

  int 
    hid, hid2d, hid_unfold, nb1, nb2, nbz, nb[2]
    ,itype, iz, i1, i2 ,NTYPE, iplot 
    ,index, ipar, IERR
    ;

  double
    x1min, x1max, x2min, x2max, xmin[2], xmax[2]
    ,x1,x2, x[2], z, w 
    ;

  //  double UNFOLD_SUM8[5];

  char 
    ctype[40]
    ,varnames[80]
    ,chis[200]
    ,ntname[20]
    ,fnam[] = "mkplots";
    ;

  // ---------- BEGIN ----------

  print_banner("Plot Results");
  plotFile = INPUTS.PLOTFILE_OUT;

  TABLEFILE_INIT();
  TABLEFILE_OPEN(plotFile, "new");


  nb1   = INPUTS.NBIN_PAR[IPAR_1]; 
  x1min = INPUTS.RANGE_PAR[IPAR_1][0] ;
  x1max = INPUTS.RANGE_PAR[IPAR_1][1] ;

  nb2   = INPUTS.NBIN_PAR[IPAR_2]; 
  x2min = INPUTS.RANGE_PAR[IPAR_2][0] ;
  x2max = INPUTS.RANGE_PAR[IPAR_2][1] ;

  nb[0] = nb1 ;
  nb[1] = nb2 ;
  xmin[0] =  x1min ;  xmin[1] = x2min ;
  xmax[0] =  x1max ;  xmax[1] = x2max ;

  nbz   = INPUTS.NBIN_PAR[IPAR_Z]; 

  NTYPE = INPUTS.NTYPE ;
  for ( itype = 1; itype <= NTYPE+2 ; itype++ ) {

    hid2d = itype ;

    if ( itype <= NTYPE ) {
      if ( INPUTS.USETYPE[itype] == 0 ) continue ;

      sprintf(ctype, "%s", INPUTS.TYPENAME[itype] );
      sprintf(varnames,"%s vs. %s"
	      ,INPUTS.PARNAME[IPAR_2][itype]
	      ,INPUTS.PARNAME[IPAR_1][itype] );
    }
    else if ( itype == NTYPE+1 ) {
      sprintf(ctype, "UNFOLDED" );
      sprintf(varnames,"%s vs. %s"
	      ,INPUTS.PARNAME[IPAR_2][ITYPE_SIMGEN]
	      ,INPUTS.PARNAME[IPAR_1][ITYPE_SIMGEN] );
    }
    else if ( itype == NTYPE+2 ) {
      sprintf(ctype, "CONTAIN-FRAC" );
      sprintf(varnames,"%s vs. %s"
	      ,INPUTS.PARNAME[IPAR_2][ITYPE_SIMFIT]
	      ,INPUTS.PARNAME[IPAR_1][ITYPE_SIMFIT] );
    }

    sprintf(chis, "%s for  %s", varnames, ctype );
    SNHIST_INIT(2, hid2d, chis, nb, xmin, xmax);

    // include 1D projections for unfolded distributions
    if ( itype == ITYPE_UNFOLD ) {
      sprintf(chis, "UNFOLDED %s", INPUTS.PARNAME[IPAR_1][ITYPE_SIMGEN] );
      hid_unfold = 50 + IPAR_1 ;
      SNHIST_INIT(1, hid_unfold, chis, &nb1, &x1min, &x1max);      

      sprintf(chis, "UNFOLDED %s", INPUTS.PARNAME[IPAR_2][ITYPE_SIMGEN] );
      hid_unfold = 50 + IPAR_2 ;
      SNHIST_INIT(1, hid_unfold, chis, &nb2, &x2min, &x2max);      
    }

    // fill histogram
    for(i1=1; i1 <= nb1; i1++ ) {
      for(i2=1; i2 <= nb2; i2++ ) {

	x1 = TABLE_BINVALUES[IPAR_1][i1]  ;
	x2 = TABLE_BINVALUES[IPAR_2][i2]  ;
	x[0] =  x1 ;
	x[1] =  x2 ;

	if ( itype <= NTYPE ) {
	  index = INDEXMAP[0][i1][i2] ;  
	  w  = TABLE[itype].ENTRIES[index] ; 
	}
	else if ( itype == NTYPE+1 ) 
	  {  w  = P_UNFOLD[i1][i2] ; }
	else if ( itype == NTYPE+2 ) 
	  {  w  = MIGRATION_TABLE[0][i1][i2].CONTAIN_FRAC ; }

	if ( w != 0.0)  
	  { SNHIST_FILL(2, hid2d, x, w ); }

	if ( itype == ITYPE_UNFOLD ) {
	  hid_unfold = 50 + IPAR_1 ; SNHIST_FILL(1,hid_unfold, &x1, w);
	  hid_unfold = 50 + IPAR_2 ; SNHIST_FILL(1,hid_unfold, &x2, w);
	}

      }
    }

  } // end of itype loop



  // plot user-requested MIGRATION matrices
  for ( iplot=1; iplot <= INPUTS.NMIGPLOT; iplot++ )
    { PLOT_MIGRATION(iplot); }


  // Now make data-prediction plots; 1-d projections only
  PLOT_DATAPRED(IPAR_Z);
  PLOT_DATAPRED(IPAR_1);
  PLOT_DATAPRED(IPAR_2);


  // print mean and RMS for each unfolded parameter

  for ( ipar=1; ipar <= INPUTS.NPAR; ipar++ ) {
    hid_unfold = 50 + ipar ;
    UNFOLDED_AVG_RMS(ipar,hid_unfold);
  }

  TABLEFILE_CLOSE(plotFile);


} // end of mkplots



// ========================
void PLOT_DATAPRED(int ipar) {

  // Make data prediction using P_UNFOLD and SIM-tables.

  int 
    nb, hid
    ,NBZ, NB1, NB2
    ,IZ,  IB1, IB2
    ,iz,  ib1, ib2
    ,NMIGBIN, imig
    ,INDEX
    ;

  double xmin, xmax, xpar[10], w ;
  char chis[100];

  double XMIG8, XGEN8, XACC8, XPRED8, XUNFOLD8 ;

  char fnam[40] = "PLOT_DATAPRED" ;

  // ------------- BEGIN -----------


  hid   = 200 + ipar;
  nb    = INPUTS.NBIN_PAR[ipar]; 
  xmin  = INPUTS.RANGE_PAR[ipar][0] ;
  xmax  = INPUTS.RANGE_PAR[ipar][1] ;

  sprintf(chis, "%s DATA prediction from UNFOLD and SIM tables", 
	  INPUTS.PARNAME[ipar][ITYPE_DATA] );

  SNHIST_INIT(1, hid, chis, &nb, &xmin ,&xmax);


  NB1 = INPUTS.NBIN_PAR[IPAR_1]; 
  NB2 = INPUTS.NBIN_PAR[IPAR_2]; 
  NBZ = INPUTS.NBIN_PAR[IPAR_Z]; // bug-fix: Jun 2011

  for ( IZ=1; IZ <= NBZ; IZ++ ) {
    for( IB1=1; IB1 <= NB1; IB1++ ) {
      for( IB2=1; IB2 <= NB2; IB2++ ) {

	INDEX = INDEXMAP[IZ][IB1][IB2] ;
	XACC8  = TABLE[ITYPE_SIMACC].ENTRIES[INDEX] ;
	XGEN8  = TABLE[ITYPE_SIMGEN].ENTRIES[INDEX] ;
	XUNFOLD8 = P_UNFOLD[IB1][IB2] ;

	if ( XGEN8 <= 0.0 ) continue ;
	if ( XACC8 <= 0.0 ) continue ;
	  
	NMIGBIN = MIGRATION_TABLE[IZ][IB1][IB2].NMIGBIN;

	for ( imig=1; imig <= NMIGBIN; imig++ ) {
	  iz    = MIGRATION_TABLE[IZ][IB1][IB2].IBINZ_NEAR[imig] ;
	  ib1   = MIGRATION_TABLE[IZ][IB1][IB2].IBIN1_NEAR[imig] ;
	  ib2   = MIGRATION_TABLE[IZ][IB1][IB2].IBIN2_NEAR[imig] ;       
	  XMIG8 = MIGRATION_TABLE[IZ][IB1][IB2].NSIMFIT_NEAR[imig];

	  xpar[IPAR_Z] = TABLE_BINVALUES[IPAR_Z][iz]  ;
	  xpar[IPAR_1] = TABLE_BINVALUES[IPAR_1][ib1]  ;
	  xpar[IPAR_2] = TABLE_BINVALUES[IPAR_2][ib2]  ;

	  XPRED8 = XMIG8 * XUNFOLD8/XGEN8 ;
	  w = XPRED8 ;
	  SNHIST_FILL(1, hid, &xpar[ipar], w );
	
	} // imig

      } // end of IB2
    }  // end of IB1
  } // IZ

} // end of PLOT_DATAPRED

// =================================
void PLOT_MIGRATION(int iplot) {

  // May 7 2013: use sntools_output 

  int IZACC, I1ACC, I2ACC, ifit ;
  double  w8, z8, par8[MAXPAR];
  char fnam[40] = "PLOT_MIGRATION" ;

  double x0min, x1min, x0max, x1max, xrad, xmin[2], xmax[2] ;
  double z, x0, x1, x[2];
  int hid, nb0, nb1, nb[2], ntmp;
  int NMIGBIN, izfit, ib1fit, ib2fit, NMIG ;
  char chis[80];

  // ----------------- BEGIN -------------

  hid = 100 + iplot;

  z8           = INPUTS.MIGVAL[iplot][IPAR_Z] ;
  par8[IPAR_1] = INPUTS.MIGVAL[iplot][IPAR_1] ;
  par8[IPAR_2] = INPUTS.MIGVAL[iplot][IPAR_2] ;

  IZACC   = PARVAL2BIN( IPAR_Z, z8    ) ;
  I1ACC   = PARVAL2BIN( IPAR_1, par8[IPAR_1] ) ;
  I2ACC   = PARVAL2BIN( IPAR_2, par8[IPAR_2] ) ;

  sprintf(chis,"Migration at z=%6.3f %s=%6.3f  %s=%6.3f", z8
	  ,INPUTS.PARNAME[IPAR_1][ITYPE_SIMACC], par8[IPAR_1]
	  ,INPUTS.PARNAME[IPAR_2][ITYPE_SIMACC], par8[IPAR_2] );

  /*
  printf(" xxx %s\n", chis);
  printf(" xxx IZ,I1,I2(ACC) = %d %d %d \n", IZACC, I1ACC, I2ACC );
  */


  ntmp = (int)(INPUTS.MIGBIN_RADIUS+0.001) + 1 ;
  nb0 = nb1 = 2*ntmp + 1 ;
  nb[0] = nb0;
  nb[1] = nb1;

  xrad  = ((double)ntmp + 0.5) * INPUTS.BINSIZE_PAR[IPAR_1] ;
  x0min = TABLE_BINVALUES[IPAR_1][I1ACC]  - xrad ;
  x0max = TABLE_BINVALUES[IPAR_1][I1ACC]  + xrad ;

  xrad  = ((double)ntmp + 0.5) * INPUTS.BINSIZE_PAR[IPAR_2] ;
  x1min = TABLE_BINVALUES[IPAR_2][I2ACC]  - xrad ;
  x1max = TABLE_BINVALUES[IPAR_2][I2ACC]  + xrad ;

  xmin[0] = x0min;  xmin[1] = x1min;
  xmax[0] = x0max;  xmax[1] = x1max;


  SNHIST_INIT(2, hid, chis, nb, xmin, xmax);

  NMIGBIN = MIGRATION_TABLE[IZACC][I1ACC][I2ACC].NMIGBIN;

  for ( ifit=1; ifit <= NMIGBIN ; ifit++ ) {
    izfit  = MIGRATION_TABLE[IZACC][I1ACC][I2ACC].IBINZ_NEAR[ifit] ;
    ib1fit = MIGRATION_TABLE[IZACC][I1ACC][I2ACC].IBIN1_NEAR[ifit] ;
    ib2fit = MIGRATION_TABLE[IZACC][I1ACC][I2ACC].IBIN2_NEAR[ifit] ;
    w8     = MIGRATION_TABLE[IZACC][I1ACC][I2ACC].NSIMFIT_NEAR[ifit];

    z  = TABLE_BINVALUES[IPAR_1][izfit]  ;
    x0 = TABLE_BINVALUES[IPAR_1][ib1fit]  ;
    x1 = TABLE_BINVALUES[IPAR_2][ib2fit]  ;
    x[0] = x0 ; x[1]=x1; 

    SNHIST_FILL(2, hid, x, w8 );

  }

} // end of PLOT_MIGRATION



// **********************************
void UNFOLDED_AVG_RMS(int ipar, int hid) {

  int  ICASE, NUM ;
  float avg, rms ;

  char cdum[20] = " " ;
  char parName[80];
  char fnam[40] = "UNFOLDED_AVG_RMS" ;

  
  // ------------- BEGIN -----------

  /*
  sprintf(parName, "%s", INPUTS.PARNAME[ipar][ITYPE_SIMGEN] );
  NUM = 1;

  ICASE = 1;  avg =  hstati_( &hid, &ICASE, cdum, &NUM, strlen(cdum) ) ;
  ICASE = 2;  rms =  hstati_( &hid, &ICASE, cdum, &NUM, strlen(cdum) ) ;
  printf("\t UNFOLDED_AVG( %-8s ): %8.3f  \n",  parName, avg );
  printf("\t UNFOLDED_RMS( %-8s ): %8.3f  \n",  parName, rms  );
  */

} // end of UNFOLDED_AVG_RMS


// *****************
void prep_bindump(void) {

  // Just print info to screen.

  int i, i1, i2, NDUMP;
  double xval[10];

  // --------- BEGIN ------

  NDUMP = INPUTS.NBINDUMP; 

  if ( NDUMP <= 0 ) return ;

  print_banner("Prepare BIN-DUMPs");

  for ( i=1; i <= NDUMP; i++ ) {
    i1 = INPUTS.IBINDUMP[i][IPAR_1] ; 
    i2 = INPUTS.IBINDUMP[i][IPAR_2] ; 
    xval[1] = TABLE_BINVALUES[IPAR_1][i1];
    xval[2] = TABLE_BINVALUES[IPAR_2][i2];

    printf(" Will dump underlying ");
    printf("%s-bin=%3.3d (%6.3f) and %s-bin=%3.3d (%6.3f) \n"
	   ,INPUTS.PARNAME[1][ITYPE_SIMGEN], i1, xval[1]
	   ,INPUTS.PARNAME[2][ITYPE_SIMGEN], i2, xval[2] 
	   );
  }

} // end of prep_bindump


// ==============================
void  TEST_PARVAL2BIN(void) {

  int i1, i;

  double av;

  // ---------- BEGIN ----------

  for ( i = -10; i < 10; i++ ) {

    av= .1*(double)i - 0.02;
    i1 = PARVAL2BIN(1, av);
    printf("  TEST_PARVAL2BIN[AV=%6.3f] = %d \n", av, i1 );
  }

  debugexit("TEST_PARVAL2BIN");

} 

// ==========================================
void DMP_ARRAYVAL(int itype, int ipar, int isn) {


    printf("  DMP_ARRAYVAL:  %s ARRAY[isn=%d]  %s = %10.4f \n"
	   ,INPUTS.TYPENAME[itype]
	   ,isn
	   ,INPUTS.PARNAME[ipar][itype]
	   ,ARRAY[itype].PAR[ipar][isn] 
	   ) ;


} // end of DMP_ARRAYVAL


// ==================================
void DMP_UNFOLD(int IZ, int IBIN1, int IBIN2, int imig ) {



  double 
    EFFIC, dP, dPSUM, PSIM 
    ,BINVAL[MAXPAR]
    ,binval[MAXPAR]
    ;

  char *CPAR[MAXPAR];

  int 
    iz, ibin1, ibin2 
    ,NSIMGEN, NSIMACC, NSIMFIT, NSIMFIT_SUM
    ,NDATA 
    ,NMIGBIN 
    ,INDEX
    ,index
    ;

  // ------------ BEGIN ----------

  iz = IZ ;

  CPAR[IPAR_Z] = INPUTS.PARNAME[IPAR_Z][ITYPE_SIMACC] ;
  CPAR[IPAR_1] = INPUTS.PARNAME[IPAR_1][ITYPE_SIMACC] ;
  CPAR[IPAR_2] = INPUTS.PARNAME[IPAR_2][ITYPE_SIMACC] ;

  BINVAL[IPAR_Z] = TABLE_BINVALUES[IPAR_Z][IZ];
  BINVAL[IPAR_1] = TABLE_BINVALUES[IPAR_1][IBIN1];
  BINVAL[IPAR_2] = TABLE_BINVALUES[IPAR_2][IBIN2];

  INDEX = INDEXMAP[IZ][IBIN1][IBIN2] ;
  NSIMACC = (int)TABLE[ITYPE_SIMACC].ENTRIES[INDEX];
  NSIMGEN = (int)TABLE[ITYPE_SIMGEN].ENTRIES[INDEX];
  NMIGBIN = MIGRATION_TABLE[IZ][IBIN1][IBIN2].NMIGBIN;

  if ( NSIMGEN > 0 ) 
    { EFFIC = (double)NSIMACC / (double)NSIMGEN ; }
  else
    { EFFIC = 0.0 ; }


  // check option to start header info for this dump to follow
  if ( imig == 0 ) {
    printf(" ---------------------------------------------------------- \n");
    printf(" Begin UNFOLD-DUMP for SIMACC values: \n");
    printf("\t %s=%6.3f(%d)  %s=%6.3f(%d)   %s=%6.3f(%d) \n"
	   ,CPAR[IPAR_Z], BINVAL[IPAR_Z], IZ
	   ,CPAR[IPAR_1], BINVAL[IPAR_1], IBIN1
	   ,CPAR[IPAR_2], BINVAL[IPAR_2], IBIN2 
	   );


    printf("\t NSIMGEN=%d   NSIMACC=%d  EFFIC=%5.3f   NMIGBIN=%d\n", 
	   NSIMGEN, NSIMACC, EFFIC, NMIGBIN );

    NSIMFIT_SUM = 0;
    dPSUM = 0.0 ;
    return ;
  }

  CPAR[IPAR_Z] = INPUTS.PARNAME[IPAR_Z][ITYPE_SIMFIT] ;
  CPAR[IPAR_1] = INPUTS.PARNAME[IPAR_1][ITYPE_SIMFIT] ;
  CPAR[IPAR_2] = INPUTS.PARNAME[IPAR_2][ITYPE_SIMFIT] ;

  binval[IPAR_Z] = TABLE_BINVALUES[IPAR_Z][iz];
  binval[IPAR_1] = TABLE_BINVALUES[IPAR_1][ibin1];
  binval[IPAR_2] = TABLE_BINVALUES[IPAR_2][ibin2];

  ibin1   = MIGRATION_TABLE[IZ][IBIN1][IBIN2].IBIN1_NEAR[imig] ;
  ibin2   = MIGRATION_TABLE[IZ][IBIN1][IBIN2].IBIN2_NEAR[imig] ;

  NSIMFIT = (int)MIGRATION_TABLE[IZ][IBIN1][IBIN2].NSIMFIT_NEAR[imig];
  index   = INDEXMAP[iz][ibin1][ibin2] ;
  NDATA   = (int)TABLE[ITYPE_DATA].ENTRIES[index] ;

  PSIM = (double)NSIMFIT / (double)NSIMACC ;
  dP   = PSIM * (double)NDATA / EFFIC ;
  dPSUM += dP;

  NSIMFIT_SUM += NSIMFIT ;
  printf("   %s=%6.3f  %s=%6.3f  NSIMFIT=%3d(SUM=%3d)  NDATA=%2d   dP=%7.3f\n"
	 ,CPAR[IPAR_1], binval[IPAR_1]
	 ,CPAR[IPAR_2], binval[IPAR_2]
	 ,NSIMFIT, NSIMFIT_SUM, NDATA , dP
	 );

  if ( imig == NMIGBIN ) {
    if (  NSIMFIT_SUM != NSIMACC ) {
      printf("\t WARNING: NSIMFIT(SUM)=%d  !=  NSIMACC=%d ***** \n", 
	     NSIMFIT_SUM, NSIMACC);
    }

    printf("\t dP_UNFOLD = %7.3f \n", dPSUM );
  }

} // end of DMP_UNFOLD
