'''
Abstract class for porcessing a set of molecular representations SMILES 
'''
class SmilesProcessor():

    '''
    Initialize SMILES's vocabulary
    '''
    def __init__(self):
        elements = 'H,He,Li,Be,B,C,N,O,F,Ne,Na,Mg,Al,Si,P,S,Cl,Ar,K,Ca,Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Ga,Ge,As,Se,Br,Kr,Rb,Sr,Y,Zr,Nb,Mo,Tc,Ru,Rh,Pd,Ag,Cd,In,Sn,Sb,Te,I,Xe,Cs,Ba,La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu,Hf,Ta,W,Re,Os,Ir,Pt,Au,Hg,Tl,Pb,Bi,Po,At,Rn,Fr,Ra,Ac,Th,Pa,U,Np,Pu,Am,Cm,Bk,Cf,Es,Fm,Md,No,Lr,Rf,Db,Sg,Bh,Hs,Mt,Ds,Rg,Cn,Uut,Fl,Uup,Lv,Uus,Uuo'
        aromatic_atoms = 'b,c,n,o,p,s,as,se,te'
        symbols = '[,],(,),=,+,-,#,:,@,.,%'
        isotopes = '0,1,2,3,4,5,6,7,8,9'

        elements = str(elements).split(',')
        aromatic_atoms = str(aromatic_atoms).split(',')
        symbols = str(symbols).split(',')
        isotopes = str(isotopes).split(',')

        # vocabulary with all the tokens in a SMILES representation 
        self.vocabulary = elements + aromatic_atoms + symbols + isotopes

    '''
    Process a single SMILES by spliting it into an array of tokens that are part of the SMILES vocabulary
    '''
    def process_smiles(self, smiles):
        tokens = []
        i = 0
        found = False
        while i < len(smiles):
          if len(smiles[i:]) >= 3:
            if smiles[i:i+3] in self.vocabulary:
              tokens.append(smiles[i:i+3])
              i += 3
              found = True
          if len(smiles[i:]) >= 2 and not found:
            if smiles[i:i+2] in self.vocabulary:
              tokens.append(smiles[i:i+2])
              i += 2
              found = True
          if len(smiles[i:]) >= 1 and not found:
            if smiles[i] in self.vocabulary:
              tokens.append(smiles[i])
              i += 1
              found = True
          if not found:
            print('Error in value', smiles[i])
            print(smiles)
            break
          found = False
        return tokens

    '''
    Process an array of SMILES into a list of processed SMILES and their respective lengths (number of tokens)
    '''
    def process_smiles_array(self, smiles_array):
        processed_list = list()
        lengths = list()
        for i in range(len(smiles_array)):
            processed_smiles = self.process_smiles(smiles_array[i])
            processed_list.append(' '.join(processed_smiles))
            lengths.append(len(processed_smiles))
        return processed_list, lengths