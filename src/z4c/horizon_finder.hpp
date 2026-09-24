#ifndef Z4C_HORIZON_FINDER_HPP_
#define Z4C_HORIZON_FINDER_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file horizon_finder.hpp
//! \brief Definition for the horizon finder interface.

#include <memory>

#include "athena.hpp"

class Driver;
class MeshBlockPack;
class ParameterInput;

class HorizonFinder {
  public:
   static std::unique_ptr<HorizonFinder> Create(MeshBlockPack *pmbp, ParameterInput *pin);
   virtual ~HorizonFinder() = default;
   virtual void Find(Driver *pdrive, int stage) = 0;
   virtual int NumHorizons() const = 0;
   virtual bool Found(int h) const = 0;
   virtual const Real *Center(int h) const = 0;
   virtual Real MinRadius(int h) const = 0;
   virtual Real Mass(int h) const = 0;
   virtual const Real *Spin(int h) const = 0;
};

#endif //Z4C_HORIZON_FINDER_HPP_