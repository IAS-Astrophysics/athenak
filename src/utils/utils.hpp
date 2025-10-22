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

#include "athena.hpp"

class MeshBlockPack;

void ShowConfig();
void ChangeRunDir(const std::string dir);
void ComputeDerivedVariable(std::string name, int index, MeshBlockPack* pmbp,
                            DvceArray5D<Real> dvars);
void WatchDog(int timeout);

#endif // UTILS_UTILS_HPP_
