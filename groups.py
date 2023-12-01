import sys
import copy
from math import factorial
import numpy as np
from numpy import exp, pi

class modadd_t:
    def __init__(self, residue, modulus):
        self.residue = residue % modulus
        self.modulus = modulus
        
    def __mul__(a,b):
        if a.modulus != b.modulus:
            print("Mixed moduli %d, %d" % (a, b))
            sys.exit(1)
        c = modadd_t(a.residue + b.residue, modulus)
        return(c)
    
    def __eq__(a,b):
        if a.residue == b.residue:
            return 1
        return 0
    
    def __ne__(a,b):
        return not a == b
    
    def __str__(self):
        return str(self.residue)
    
    def inv(a):
        c = modadd_t(-a.residue, a.modulus)
        return c 

class v4:
    def __init__(self, argcode):
        self.code = argcode & 3
    
    def __eq__(a,b):
        if a.code == b.code:
            return 1
        return 0
    
    def __ne__(a,b):
        return not a == b
    
    def __mul__(a,b):
        c = v4(a.code ^ b.code)
        return(c)
    
    def inv(a):
        c = v4(a.code)
        return c
    
    def __str__(self):
        if self.code == 0:
            return "e"
        
        elif self.code == 1:
            return "a"
        
        elif self.code == 2:
            return "b"
        
        elif self.code == 3:
            return "c"
        
        else:
            raise IOError
            
class dih:
    def __init__(self, argrot, argflip, argn):
        self.n = argn
        self.rot = argrot % self.n
        self.flip = argflip & 1
    
    def __eq__(a,b):
        if a.rot == b.rot and a.flip == b.flip:
            return 1
        return 0
    
    def __ne__(a,b):
        return not a == b
    
    def __mul__(a,b):
        if a.n != b.n:
            raise RuntimeError
        elif a.flip:
            crot = a.rot-b.rot
        else:
            crot = a.rot+b.rot
        c = dih(crot, a.flip^b.flip, a.n)
        return c
    
    def inv(a):
        if a.flip:
            c = dih(a.rot, a.flip, a.n)
            return c
        else:
            c = dih(-a.rot, a.flip, a.n)
            return c
        
    def __str__(self):
        return str(self.rot)+","+str(self.flip)
    
    def __hash__(self):
        return hash((self.rot, self.flip, self.n))
    
    def power(a, m):
        if a.flip:
            if m % 2:
                c = dih(a.rot, a.flip, a.n)
            else:
                c = dih(0,0,a.n)
        else:
            crot = m*a.rot
            c = dih(crot, a.flip, a.n)
        return c
    
    @classmethod
    def get_elements(cls, n):
        elts = []
        for i in range(n):
            for j in range(2):
                elt = dih(i,j,n)
                elts.append(elt)
        return(elts)
    
    def conjugacy_class(a):
        Group = dih.get_elements(a.n)
        conjugancy_class_set = []
        for c in Group:
            if c.inv()*a*c not in conjugancy_class_set:
                conjugancy_class_set.append(c.inv()*a*c)
        return(conjugancy_class_set)
    
    def Z(a):
        Group = dih.get_elements(a.n)
        Center = []
        for c in Group:
            if c*a == a*c and c not in Center:
                Center.append(c)
        return Center

    def Z_name(a):
        Center = a.Z()
        if a.n != 4:
            print("Zenter subgroup names only available for D4")
            raise RuntimeError

        if len(Center) == 8:
            output = "dihedral"
        else:
            output = "cyclic"
            for c in Center:
                if c.flip == 1:
                    output = "klein four"
        return(output)
        
    def get_irreps(a):
        #This get's the irreps of the center of the arguement.
        if a.n != 4:
            print("Irreducible representations only available for D4")
            raise RuntimeError
        GroupName = a.Z_name()
        Group = a.Z()
        if GroupName == "dihedral":
            def A(g):
                return(1)
            def B(g):
                return((-1)**(g.flip))
            def C(g):
                return((-1)**(g.rot))
            def D(g):
                return((-1)**(g.rot+g.flip))
            def E(g):
                Rot = np.linalg.matrix_power(np.array([[0, -1], [1, 0]]), g.rot)
                Flip = np.linalg.matrix_power(np.array([[1,0], [0,-1]]), g.flip)
                output = Rot@Flip
                return(output)
            return([A, B, C, D, E])
        elif GroupName == "cyclic":
            def A(g):
                return(1)
            def B(g):
                return(exp(1j*g.rot*pi))
            def C(g):
                return(exp(1j*g.rot*pi/2))
            def D(g):
                return(exp(-1j*g.rot*pi))
            return([A,B,C,D])
        else:
            if dih(2,1,4) in Group:
                def A(g):
                    return(1)
                def B(g):
                    return((-1)**(g.flip))
                def C(g):
                    return((-1)**(g.rot/2))
                def D(g):
                    return((-1)**(g.rot/2+g.flip))
            else:
                def A(g):
                    return(1)
                def B(g):
                    return((-1)**(g.flip))
                def C(g):
                    return((-1)**(g.flip))
                def D(g):
                    return((-1)**(g.flip+g.rot))
            return([A,B,C,D])
        
    def q_representative(r,c):
        Group = dih.get_elements(r.n)
        if c not in r.conjugacy_class():
            print("c not in C(r)")
            raise RuntimeError
        for g in Group:
            if c == g*r*(g.inv()):
                return(g)
    
class perm:
    def __init__(self, images, n):
        if len(images) != n:
            raise RuntimeError
        self.n = n
        self.images = copy.copy(images)
    def __mul__(a,b):
        if a.n != b.n:
            raise RuntimeError
        c = perm(list(range(a.n)), a.n)
        for i in range(a.n):
            c.images[i] = a.images[b.images[i]]
        return c
    def check_permutation(self):
        test = copy.copy(self.images)
        test.sort()
        for i in range(self.n):
            if test[i] != i:
                print ("Not a permutation:",self.images)
                print("Test:", test)
                raise RuntimeError
    def inv(a):
        c = perm(list(range(a.n)), a.n)
        for i in range(a.n):
            c.images[a.images[i]] = i
        return c
    def parity(self):
        nswap = 0
        n = self.n
        imsort = copy.copy(self.images)
        top = n-1
        while top > 0:
            for i in range(top):
                if imsort[i] > imsort[i+1]:
                    temp = imsort[i]
                    imsort[i+1] = temp
                    nswap += 1
            top = top-1
        return(nswap & 1)
    def sgn(self):
        if self.parity() == 0:
            return 1
        else:
            return -1
    def kth_perm(k, n, nfact):
        nifact = nfact
        images = list(range(n))
        temp = list(range(n+1))
        
        ni = n
        for pos in range(n):
            nifact /= ni
            r = k % nifact
            q = int(k/nifact)
            k = r
            
            images[pos] = temp[q]+1
            for i in range(q,ni):
                temp[i] = temp[i+1]
            ni = ni-1
        return perm(images, n)
    @classmethod
    def get_elements(cls,n):
        group_size = factorial(n)
        elts = []
        for k in range(group_size):
            elt = perm.kth_perm(k, n, group_size)
            elts.append(elt)
            
class DrinfeldDouble:
    Group = dih.get_elements(4)
    basis = []
    for g in Group:
        for h in Group:
            basis.append((g,h))
            
    def __init__(self, data):
        if type(data) == list:
            data_set = set({})
            for entry in data:
                coeff, basis_element = entry
                i, j = basis_element
                basis_index = 8*i+j
                data_set.add((coeff, DrinfeldDouble.basis[basis_index]))
            self.data = data_set
        else:
            self.data = data
        
    def __eq__(a,b):
        if a.data == b.data:
            return True
        
    def __add__(a,b):
        c_data = set({})
        temp_data = a.data.union(b.data)
        for term1 in temp_data:
            for term2 in temp_data:
                if term1 == term2:
                    basis_element = term1[1]
                    coeff = term1[0]+term2[0]
                    c_data.add((coeff, basis_element))
        c = DrinfeldDouble(c_data)
        return(c)
    
    def scalar_mul(a,b):
        c_data = set({})
        for term in a.data:
            coeff = b*term[0]
            c_data.add((coeff, term[1]))
        c = DrinfeldDouble(c_data)
        return(c)
    
    def __mul__(a,b):
        c = DrinfeldDouble(set({}))
        for a_term in a.data:
            for b_term in b.data:
                coeff = a_term[0]*b_term[0]
                g1, h1 = a_term[1]
                g2, h2 = b_term[1]
                basis_element = (g1*g2, dih(0,0,4))
                if h1 == h2:
                    basis_element = (g1*g2, h1)
                c += DrinfeldDouble(set({(coeff, basis_element)}))
        return(c.scalar_mul(1/2))
    @classmethod
    def get_irreps(cls):
        irreps = set({})
        
        
        
            
            
            
            
            
            
            

