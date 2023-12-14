#ifndef UTILS_UTILS_HPP_
#define UTILS_UTILS_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file utils.hpp
//  \brief prototypes of functions in utils/*.cpp files
//  These "utility" functions provide a variety of useful features.

#include <string>

void ShowConfig();
void ChangeRunDir(const std::string dir);
int CreateMPITag(int lid, int buff_id, int phys_id);

#endif // UTILS_UTILS_HPP_
