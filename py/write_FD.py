import sys

def string_central(stencil,a=0,b=0):
    a_ = stencil.split(", ")
    len_stenc = len(a_)
    mid_stenc = len_stenc//2
    string_der = ""
    for i, s in enumerate(a_):
        if s == "0.":
            continue
        elif not s[0] == "-":
            s = "+" + s
        if a:
            if b:
                aux = "{:^10} * quant(m,a,b".format(s)
            else:
                aux = "{:^10} * quant(m,a".format(s)
        else:
            aux = "{:^10} * quant(m".format(s)
        offset = " " * (len(aux) + len("    case 2 : out =") + 1)
        fact = i-mid_stenc
        if fact == 0:
            string_der += aux + ",k,\n{offset}j,\n{offset}i)\n".format(offset = offset) + " " * (len("    case 2 : out ="))
        else:
            string_der += aux + ",k+({fact:>2})*shiftk,\n{offset}j+({fact:>2})*shiftj,\n{offset}i+({fact:>2})*shifti)\n".format(fact = str(fact), offset = offset)+ " " * (len("    case 2 : out ="))
    return string_der.strip()+";"

def string_mixed(stencil,a=0,b=0):
    a_ = stencil.split(", ")
    len_stenc = len(a_)
    mid_stenc = len_stenc//2
    string_der = ""
    for i, si in enumerate(a_):
        for j, sj in enumerate(a_):
          if si == "0." or sj == "0.":
              continue
          if a:
              if b:
                  aux = "+({:^10}) * ({:^10}) * quant(m,a,b".format(si, sj)
              else:
                  aux = "+({:^10}) * ({:^10}) * quant(m,a".format(si, sj)
          else:
              aux = "+({:^10}) * ({:^10}) * quant(m".format(si, sj)
          offset = " " * (len(aux) + 1 + len("    case 2 : out ="))
          string_der += aux + ",k+({fact1:>2})*shiftxk + ({fact2:>2})*shiftyk,\n{offset}j+({fact1:>2})*shiftxj + ({fact2:>2})*shiftyj,\n{offset}i+({fact1:>2})*shiftxi + ({fact2:>2})*shiftyi)\n".format(fact1 = i-mid_stenc, fact2 = j-mid_stenc, offset = offset) + " " * (len("    case 2 : out ="))
    return string_der.strip()+";"

def string_biased(stencil,left=1,a=0,b=0):
    label = "Left" if left else "Right"
    a_ = stencil.split(", ")
    len_stenc = len(a_)
    if len_stenc == 3:
        biased = 0
    elif len_stenc == 5:
        biased = 1
    elif len_stenc == 7:
        biased = 2
    if left:
        mid_stenc = len_stenc-1-biased
    else:
        mid_stenc = -(len_stenc-1)+biased
        for i, s in enumerate(a_):
            if s[0] == "-":
                a_[i] = s[1::]
            else:
                a_[i] = "-" + s
    for i, s in enumerate(a_):
        if s[0] == "-":
            pass
        elif not s[0] == "+":
            a_[i] = "+" + s
    string_der = ""
    for i, s in enumerate(a_):
        if s == "0.":
            continue
        if a:
            if b:
                aux = "{:^10} * quant(m,b,c".format(s)
            else:
                aux = "{:^10} * quant(m,b".format(s)
        else:
            aux = "{:^10} * quant(m".format(s)
        if left:
            fact = i-mid_stenc
        else:
            fact = -mid_stenc-i
        offset = " " * (len(aux) + 1 + len("    case 2 : out ="))
        if fact == 0:
            string_der += aux + ",k,\n{offset}j,\n{offset}i)\n".format(offset = offset) + " " * (len("    case 2 : out =")) 
        else:
            string_der += aux + ",k+({fact})*shiftk,\n{offset}j+({fact})*shiftj,\n{offset}i+({fact})*shifti)\n".format(fact = fact, offset = offset) + " " * (len("    case 2 : out ="))
    return string_der.strip()+";"


def stencils(stencil,mixed=0,biased=0):
    if mixed:
        string_mixed(stencil)
        string_mixed(stencil,1)
        string_mixed(stencil,1,1)
    elif biased:
        string_biased(stencil,left=1)
        string_biased(stencil,left=1,a=1)
        string_biased(stencil,left=1,a=1,b=1)
        string_biased(stencil,left=0)
        string_biased(stencil,left=0,a=1)
        string_biased(stencil,left=0,a=1,b=1)
    else:
        string_central(stencil)
        string_central(stencil,1)
        string_central(stencil,1,1)


def print_diff():
    # 1st derivative stencils

    # 2 ghosts
    string = "-1./2., 0., 1./2."
    stencils(string)
    stencils(string,1)
    
    # 3 ghosts
    string = "1./12., -2./3., 0., 2./3., -1./12."
    str_der = string_central(string)
    stencils(string)
    stencils(string,1)
    
    # 4 ghosts
    string = "-1./60., 3./20., -3./4., 0., 3./4., -3./20., 1./60."
    str_der = string_central(string)
    stencils(string)
    stencils(string,1)
    exit()
    
    # 2nd derivative stencils
    
    # 2 ghosts
    string = "1., -2., 1."
    stencils(string)
    
    # 3 ghosts
    string = "-1./12., 4./3., -5./2., 4./3., -1./12."
    stencils(string)
    
    # 4 ghosts
    string = "1./90., -3./20., 3./2., -49./18., 3./2., -3./20., 1./90."
    stencils(string)

    # Biased stencils
    
    # 2 ghosts
    # Left biased coefficients are the reversed of - right biased coefficients
    string = "0.5, -2.0, 1.5" #Left Biased
    stencils(string,mixed=0,biased=1)
    stencils(string,mixed=0,biased=1)
    
    string = "-1./12., 6./12., -18./12., 10./12., 3./12."
    stencils(string,mixed=0,biased=1)
    stencils(string,mixed=0,biased=1)
    
    string = "1./60., -2./15., 1./2., -4./3., 7./12., 2./5., -1./30."
    stencils(string,mixed=0,biased=1)
    stencils(string,mixed=0,biased=1)

def generate_central(strings,name_diff,a=0,b=0):
    indices = ""
    label = "scalar"
    if a == 1:
        indices += " int const a,"
        label = "vector"
        if b == 1:
            indices += " int const b,"
            label = "2D tensor"
    if name_diff == "Dx" or name_diff == "Dxx" or name_diff == "Diss":
        dx_2 = string_central(strings[0],a,b)
        dx_3 = string_central(strings[1],a,b)
        dx_4 = string_central(strings[2],a,b)
        first_line = "Real {}(int const dir, int const nghost,".format(name_diff)      
        shifts = """int const shiftk = dir==2;
  int const shiftj = dir==1;
  int const shifti = dir==0;"""
        idx = "idx[dir]"
        if name_diff == "Dxx":
            idx += "*idx[dir]"
    else:
        dx_2 = string_mixed(strings[0],a,b)
        dx_3 = string_mixed(strings[1],a,b)
        dx_4 = string_mixed(strings[2],a,b)
        first_line = "Real {}(int const dirx, int const diry, int const nghost,".format(name_diff)
        shifts = """int const shiftxk = dirx==2;
  int const shiftxj = dirx==1;
  int const shiftxi = dirx==0;
  int const shiftyk = diry==2;
  int const shiftyj = diry==1;
  int const shiftyi = diry==0;"""
        idx = "idx[dirx]*idx[diry]"

    central_code =\
"""
  // Reminder: this code has been generated with py/{}, please do modifications there. 
  // 1st derivative {}
  template <typename TYPE>
KOKKOS_INLINE_FUNCTION
  {}
          const Real idx[], const TYPE &quant,
          int const m,{}
          int const k, int const j, int const i)
  {{
  {}
  Real out;
  switch(nghost) {{
    case 2 : out = {}
             break;
    case 3 : out = {}
             break;
    case 4 : out = {}
             break;
  }}
  return out*{};
  }}\n
"""
    return central_code.format(sys.argv[0], label, first_line, indices, shifts, dx_2, dx_3, dx_4, idx)


def generate_biased(strings,name_diff,a=0,b=0):
    indices = ""
    indices_2 = ""
    label = "scalar"
    if a == 1:
        indices += " int const b,"
        indices_2 += "b,"
        label = "vector"
        if b == 1:
            indices += " int const c,"
            indices_2 += "c,"
            label = "2D tensor"
    dx_2_l = string_biased(strings[0],1,a,b)
    dx_3_l = string_biased(strings[1],1,a,b)
    dx_4_l = string_biased(strings[2],1,a,b)
    dx_2_r = string_biased(strings[0],0,a,b)
    dx_3_r = string_biased(strings[1],0,a,b)
    dx_4_r = string_biased(strings[2],0,a,b)
    shifts = """int const shiftk = dir==2;
  int const shiftj = dir==1;
  int const shifti = dir==0;"""

    central_code =\
"""
  // Reminder: this code has been generated with py/{}, please do modifications there.
  // 1st advective derivative {}
  template <typename TYPE1, typename TYPE2>
KOKKOS_INLINE_FUNCTION 
  Real Lx(int const dir,    int const nghost,
          const Real idx[], const TYPE1 &vx, const TYPE2 &quant,
          int const m, int const a,{}
          int const k, int const j, int const i)
  {{
  {}
  Real dl, dr;
  switch(nghost) {{
    case 2 : dl = {}
             dr = {}
             break;
    case 3 : dl = {}
             dr = {}
             break;
    case 4 : dl = {}
             dr = {}
             break;
  }}
  return ((vx(m,a,k,j,i) < 0) ? (vx(m,a,k,j,i) * dl) : (vx(m,a,k,j,i) * dr)) * idx[dir];
  }}\n
"""
    return central_code.format(sys.argv[0], label, indices, shifts, dx_2_l, dx_2_r, dx_3_l, dx_3_r, dx_4_l, dx_4_r, indices_2, indices_2, indices_2)



if __name__=="__main__":
    f = open("z4c_template.hpp", "r")
    template = f.read()
    f.close()

    Dx_strings = ["-1./2., 0., 1./2.", "1./12., -2./3., 0., 2./3., -1./12.", "-1./60., 3./20., -3./4., 0., 3./4., -3./20., 1./60."]
    Dx = generate_central(Dx_strings,"Dx") + generate_central(Dx_strings,"Dx",a=1) + generate_central(Dx_strings,"Dx",a=1,b=1)

    Dxx_strings = ["1., -2., 1.", "-1./12., 4./3., -5./2., 4./3., -1./12.",  "1./90., -3./20., 3./2., -49./18., 3./2., -3./20., 1./90."]
    
    Dxx = generate_central(Dxx_strings,"Dxx") + generate_central(Dxx_strings,"Dxx",a=1) + generate_central(Dxx_strings,"Dxx",a=1,b=1)
    
    Dxy = generate_central(Dx_strings,"Dxy") + generate_central(Dx_strings,"Dxy",a=1) + generate_central(Dx_strings,"Dxy",a=1,b=1)
    
    Lx_strings = ["0.5, -2.0, 1.5", "-1./12., 6./12., -18./12., 10./12., 3./12.", "1./60., -2./15., 1./2., -4./3., 7./12., 2./5., -1./30."]
    Lx = generate_biased(Lx_strings,"Lx") +  generate_biased(Lx_strings,"Lx",1) + generate_biased(Lx_strings,"Lx",1,1)

    Diss_string = ["1., -4., 6., -4., 1.", "1., -6., 15., -20., 15., -6., 1.", "1., -8., 28., -56., 70., -56., 28., -8., 1."]
    Diss = generate_central(Diss_string,"Diss",a=1)

    ders = Dx + Dxx + Dxy + Lx + Diss

    z4c_hpp = template.format(sys.argv[0], derivatives=ders)
    f = open("z4c.hpp", "w")
    f.write(z4c_hpp)
    f.close()


