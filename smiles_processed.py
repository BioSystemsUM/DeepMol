import numpy as np
import pandas as pd

class smile_array():
  def __init__(self):

      self.elements = 'H,He,Li,Be,B,C,N,O,F,Ne,Na,Mg,Al,Si,P,S,Cl,Ar,K,Ca,Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Ga,Ge,As,Se,Br,Kr,Rb,Sr,Y,Zr,Nb,Mo,Tc,Ru,Rh,Pd,Ag,Cd,In,Sn,Sb,Te,I,Xe,Cs,Ba,La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu,Hf,Ta,W,Re,Os,Ir,Pt,Au,Hg,Tl,Pb,Bi,Po,At,Rn,Fr,Ra,Ac,Th,Pa,U,Np,Pu,Am,Cm,Bk,Cf,Es,Fm,Md,No,Lr,Rf,Db,Sg,Bh,Hs,Mt,Ds,Rg,Cn,Uut,Fl,Uup,Lv,Uus,Uuo'
      self.aromatic_atoms = 'b,c,n,o,p,s,as,se,te'
      self.symbols = '[,],(,),=,+,-,#,:,@,.,%'
      self.isotopes = '0,1,2,3,4,5,6,7,8,9'

      self.elements = str(self.elements).split(',')
      self.aromatic_atoms = str(self.aromatic_atoms).split(',')
      self.symbols = str(self.symbols).split(',')
      self.isotopes = str(self.isotopes).split(',')

      self.smiles_vocabulary = self.elements + self.aromatic_atoms + self.symbols + self.isotopes


  def process_smiles(self, smiles, vocabulary):
    tokens = []
    i = 0;
    found = False;
    while i < len(smiles):
      if len(smiles[i:]) >= 3:
        if smiles[i:i+3] in vocabulary:
          tokens.append(smiles[i:i+3])
          i += 3
          found = True
      if len(smiles[i:]) >= 2 and not found:
        if smiles[i:i+2] in vocabulary:
          tokens.append(smiles[i:i+2])
          i += 2
          found = True
      if len(smiles[i:]) >= 1 and not found:
        if smiles[i] in vocabulary:
          tokens.append(smiles[i])
          i += 1
          found = True
      if not found:
        print('Error in value', smiles[i])
        print(smiles)
        break
      found = False
    return tokens

  def process_smiles_array(self, smiles_array):
    processed_list = list()
    lengths = list()
    for i in range(len(smiles_array)):
        processed_smiles = self.process_smiles(smiles_array[i], self.smiles_vocabulary)
        processed_list.append(' '.join(processed_smiles))
        lengths.append(len(processed_smiles))
    return processed_list, lengths