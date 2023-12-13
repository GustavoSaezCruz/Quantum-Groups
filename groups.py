import sys
import copy
from math import factorial
import numpy as np
from numpy import exp, pi

class modadd_t:
    def __init__(self, residue, modulus):
        self.residue = residue % modulus
        self.modulus = modulus
        
    def __mul__(self, b):
        if self.modulus != b.modulus:
            print("Mixed moduli %d, %d" % (self, b))
            sys.exit(1)
        c = modadd_t(self.residue + b.residue, self.modulus)
        return c
    
    def __eq__(self, b):
        return self.residue == b.residue
    
    def __ne__(self, b):
        return not self == b
    
    def __str__(self):
        return str(self.residue)
    
    def inv(self):
        c = modadd_t(-self.residue, self.modulus)
        return c 

class v4:
    def __init__(self, argcode):
        self.code = argcode & 3
    
    def __eq__(self, b):
        return self.code == b.code
    
    def __ne__(self, b):
        return not self == b
    
    def __mul__(self, b):
        c = v4(self.code ^ b.code)
        return c
    
    def inv(self):
        c = v4(self.code)
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
    """
    Represents a dihedral group element.

    Args:
        argrot (int): The rotation value.
        argflip (int): The flip value.
        argn (int): The order of the group.

    Attributes:
        n (int): The order of the group.
        rot (int): The rotation value modulo n.
        flip (int): The flip value modulo 2.

    Methods:
        __eq__(self, b): Checks if two dihedral group elements are equal.
        __ne__(self, b): Checks if two dihedral group elements are not equal.
        __mul__(self, b): Multiplies two dihedral group elements.
        inv(self): Returns the inverse of the dihedral group element.
        __str__(self): Returns a string representation of the dihedral group element.
        __hash__(self): Returns the hash value of the dihedral group element.
        power(self, m): Raises the dihedral group element to a power.
        get_elements(cls, n): Returns a list of all dihedral group elements for a given order.
        conjugacy_class(self): Returns the conjugacy class of the dihedral group element.
        Z(self): Returns the center of the dihedral group.
        Z_name(self): Returns the name of the center subgroup.
        get_irreps(self): Returns a list of irreducible representations of the dihedral group.
        q_representative(self, c): Returns a representative of the conjugacy class c.
    """
    def __init__(self, argrot, argflip, argn):
        self.n = argn
        self.rot = argrot % self.n
        self.flip = argflip & 1
    
    def __eq__(self, b):
        return self.rot == b.rot and self.flip == b.flip
    
    def __ne__(self, b):
        return not self == b
    
    def __mul__(self, b):
        if self.n != b.n:
            raise RuntimeError
        elif self.flip:
            crot = self.rot - b.rot
        else:
            crot = self.rot + b.rot
        c = dih(crot, self.flip ^ b.flip, self.n)
        return c
    
    def inv(self):
        if self.flip:
            c = dih(self.rot, self.flip, self.n)
            return c
        else:
            c = dih(-self.rot, self.flip, self.n)
            return c
        
    def __str__(self):
        return str(self.rot) + "," + str(self.flip)
    
    def __hash__(self):
        return hash((self.rot, self.flip, self.n))
    
    def power(self, m):
        if self.flip:
            if m % 2:
                c = dih(self.rot, self.flip, self.n)
            else:
                c = dih(0, 0, self.n)
        else:
            crot = m * self.rot
            c = dih(crot, self.flip, self.n)
        return c
    
    @classmethod
    def get_elements(cls, n):
        elts = []
        for i in range(n):
            for j in range(2):
                elt = dih(i, j, n)
                elts.append(elt)
        return elts
    
    def conjugacy_class(self):
        Group = dih.get_elements(self.n)
        conjugancy_class_set = []
        for c in Group:
            if c.inv() * self * c not in conjugancy_class_set:
                conjugancy_class_set.append(c.inv() * self * c)
        return conjugancy_class_set
    
    def Z(self):
        Group = dih.get_elements(self.n)
        Center = []
        for c in Group:
            if c * self == self * c and c not in Center:
                Center.append(c)
        return Center

    def Z_name(self):
        Center = self.Z()
        if self.n != 4:
            print("Zenter subgroup names only available for D4")
            raise RuntimeError

        if len(Center) == 8:
            output = "dihedral"
        else:
            output = "cyclic"
            for c in Center:
                if c.flip == 1:
                    output = "klein four"
        return output
        
    def get_irreps(self):
        if self.n != 4:
            print("Irreducible representations only available for D4")
            raise RuntimeError
        GroupName = self.Z_name()
        Group = self.Z()
        if GroupName == "dihedral":
            def A(g):
                return 1
            def B(g):
                return (-1) ** (g.flip)
            def C(g):
                return (-1) ** (g.rot)
            def D(g):
                return (-1) ** (g.rot + g.flip)
            def E(g):
                Rot = np.linalg.matrix_power(np.array([[0, -1], [1, 0]]), g.rot)
                Flip = np.linalg.matrix_power(np.array([[1, 0], [0, -1]]), g.flip)
                output = Rot @ Flip
                return output
            return [A, B, C, D, E]
        elif GroupName == "cyclic":
            def A(g):
                return 1
            def B(g):
                return exp(1j * g.rot * pi)
            def C(g):
                return exp(1j * g.rot * pi / 2)
            def D(g):
                return exp(-1j * g.rot * pi)
            return [A, B, C, D]
        else:
            if dih(2, 1, 4) in Group:
                def A(g):
                    return 1
                def B(g):
                    return (-1) ** (g.flip)
                def C(g):
                    return (-1) ** (g.rot / 2)
                def D(g):
                    return (-1) ** (g.rot / 2 + g.flip)
            else:
                def A(g):
                    return 1
                def B(g):
                    return (-1) ** (g.flip)
                def C(g):
                    return (-1) ** (g.flip)
                def D(g):
                    return (-1) ** (g.flip + g.rot)
            return [A, B, C, D]
        
    def q_representative(self, c):
        Group = dih.get_elements(self.n)
        if c not in self.conjugacy_class():
            print("c not in C(r)")
            raise RuntimeError
        for g in Group:
            if c == g * self * (g.inv()):
                return g
    
class perm:
    def __init__(self, images, n):
        if len(images) != n:
            raise RuntimeError
        self.n = n
        self.images = copy.copy(images)
        
    def __mul__(self, b):
        if self.n != b.n:
            raise RuntimeError
        c = perm(list(range(self.n)), self.n)
        for i in range(self.n):
            c.images[i] = self.images[b.images[i]]
        return c
    
    def check_permutation(self):
        test = copy.copy(self.images)
        test.sort()
        for i in range(self.n):
            if test[i] != i:
                print("Not a permutation:", self.images)
                print("Test:", test)
                raise RuntimeError
    
    def inv(self):
        c = perm(list(range(self.n)), self.n)
        for i in range(self.n):
            c.images[self.images[i]] = i
        return c
    
    def parity(self):
        nswap = 0
        n = self.n
        imsort = copy.copy(self.images)
        top = n - 1
        while top > 0:
            for i in range(top):
                if imsort[i] > imsort[i + 1]:
                    temp = imsort[i]
                    imsort[i + 1] = temp
                    nswap += 1
            top = top - 1
        return nswap & 1
    
    def sgn(self):
        if self.parity() == 0:
            return 1
        else:
            return -1
    
    @classmethod
    def kth_perm(cls, k, n, nfact):
        nifact = nfact
        images = list(range(n))
        temp = list(range(n + 1))
        
        ni = n
        for pos in range(n):
            nifact /= ni
            r = k % nifact
            q = int(k / nifact)
            k = r
            
            images[pos] = temp[q] + 1
            for i in range(q, ni):
                temp[i] = temp[i + 1]
            ni = ni - 1
        return perm(images, n)
    
    @classmethod
    def get_elements(cls, n):
        group_size = factorial(n)
        elts = []
        for k in range(group_size):
            elt = perm.kth_perm(k, n, group_size)
            elts.append(elt)
        return elts
            
class DrinfeldDouble:
    Group = dih.get_elements(4)
    basis = []
    for g in Group:
        for h in Group:
            basis.append((g, h))
            
    def __init__(self, data):
        if type(data) == list:
            data_set = set({})
            for entry in data:
                coeff, basis_element = entry
                i, j = basis_element
                basis_index = 8 * i + j
                data_set.add((coeff, DrinfeldDouble.basis[basis_index]))
            self.data = data_set
        else:
            self.data = data
        
    def __eq__(self, b):
        return self.data == b.data
        
    def __add__(self, b):
        c_data = set({})
        temp_data = self.data.union(b.data)
        for term1 in temp_data:
            for term2 in temp_data:
                if term1 == term2:
                    basis_element = term1[1]
                    coeff = term1[0] + term2[0]
                    c_data.add((coeff, basis_element))
        c = DrinfeldDouble(c_data)
        return c
    
    def scalar_mul(self, b):
        c_data = set({})
        for term in self.data:
            coeff = b * term[0]
            c_data.add((coeff, term[1]))
        c = DrinfeldDouble(c_data)
        return c
    
    def __mul__(self, b):
        c = DrinfeldDouble(set({}))
        for a_term in self.data:
            for b_term in b.data:
                coeff = a_term[0] * b_term[0]
                g1, h1 = a_term[1]
                g2, h2 = b_term[1]
                basis_element = (g1 * g2, dih(0, 0, 4))
                if h1 == h2:
                    basis_element = (g1 * g2, h1)
                c += DrinfeldDouble(set({(coeff, basis_element)}))
        return c.scalar_mul(1 / 2)
    
    # @classmethod
    # def get_irreps(cls):
    #     irreps = set({})
    #     return irreps
