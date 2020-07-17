//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file change_rundir.cpp
//! \brief executes unix 'chdir' command to change dir in which Athena++ runs

#include <sys/stat.h>  // mkdir()
#include <unistd.h>    // chdir()
#include <iostream>

#include "athena.hpp"

//----------------------------------------------------------------------------------------
//! \fn void ChangeRunDir(const char *pdir)
//  \brief change to input run directory; create if it does not exist yet

void ChangeRunDir(const char *pdir) {

  if (pdir == nullptr || *pdir == '\0') return;

  mkdir(pdir, 0775);
  if (chdir(pdir)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
        << "Cannot cd to directory '" << pdir << "'";
    exit(EXIT_FAILURE);
  }

  return;
}
