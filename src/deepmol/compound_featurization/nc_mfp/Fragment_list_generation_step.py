from rdkit import Chem
from rdkit.Chem import AllChem
import itertools


class Fragment_List_Generation:

    #Generate fragments of query compounds
    def generate_qMol_Fragment_List(self, All_Scaffold_FilePath, qMol_Smarts):

        Final_qMol_Fragment_List = []
        S_all_Smarts = []

        f=open(All_Scaffold_FilePath, 'r')

        while True:
            line = f.readline()
            if not line:break
            new_line = line.split("\t")
            S_all_Smarts.append(new_line[1])

        for ai in range (0, len(S_all_Smarts)):
            qMol = Chem.MolFromSmarts(qMol_Smarts)
            if qMol.HasSubstructMatch(Chem.MolFromSmarts(S_all_Smarts[ai])):
                rm = AllChem.DeleteSubstructs(qMol, Chem.MolFromSmarts(S_all_Smarts[ai]))
                Frags = str(Chem.MolToSmarts(rm)).split(".")
                Final_qMol_Fragment_List.append(Frags)

        #Add fragment lists
        Final_qMol_Fragment_List = list(itertools.chain(*Final_qMol_Fragment_List))

        #Remove duplicated fragments
        Final_qMol_Fragment_List = list(set(Final_qMol_Fragment_List))

        return Final_qMol_Fragment_List

    #Generate fragments of all query compounds
    def generate_all_Fragment_List(self, All_Scaffold_FilePath, qMol_Smarts):

        Final_all_Fragment_List = []
        S_all_Smarts = []

        f=open(All_Scaffold_FilePath, 'r')

        while True:
            line = f.readline()
            if not line:break
            new_line = line.split("\t")
            S_all_Smarts.append(new_line[1])

        for ai in range (0, len(S_all_Smarts)):
            qMol = Chem.MolFromSmarts(qMol_Smarts)
            if qMol.HasSubstructMatch(Chem.MolFromSmarts(S_all_Smarts[ai])):
                rm = AllChem.DeleteSubstructs(qMol, Chem.MolFromSmarts(S_all_Smarts[ai]))
                Frags = str(Chem.MolToSmarts(rm)).split(".")
                Final_all_Fragment_List.append(Frags)

        #Add fragment lists
        Final_all_Fragment_List = list(itertools.chain(*Final_all_Fragment_List))

        #Remove duplicated fragments
        Final_all_Fragment_List = list(set(Final_all_Fragment_List))

        return Final_all_Fragment_List

    #Generate fragment dictionary of all query compounds
    def generate_all_Fragment_Dictionary(self, Final_all_Fragment_List):

        Final_all_Fragment_Dic = {}

        Final_all_Fragment_List.sort()

        for ai in range(0, len(Final_all_Fragment_List)):
            Final_all_Fragment_Dic[Final_all_Fragment_List[ai]] = "F"+str(ai+1)

        return Final_all_Fragment_Dic




