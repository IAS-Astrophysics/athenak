#include <cstdio>
#include <initializer_list>
#include "globals.hpp"
int main() {
  for (int limit : {32767,262143,1048575,2147483647}) {
    int bits=global_variable::MPILocalIDBits(limit,14);
    if (((1ULL<<(bits+6))-1)>static_cast<unsigned long long>(limit)) return 1;
    if (bits<14 && ((1ULL<<(bits+7))-1)<=static_cast<unsigned long long>(limit)) return 2;
    int max_lid=(1<<bits)-1;
    for (int buffer=0; buffer<64; ++buffer) {
      int tag=(buffer<<bits)|max_lid;
      if ((tag>>bits)!=buffer || (tag&max_lid)!=max_lid || tag>limit) return 3;
    }
  }
  if (global_variable::MPILocalIDBits(262143,14)!=12) return 4;
  std::puts("MPI tag-layout limits and encoding passed.");
}
