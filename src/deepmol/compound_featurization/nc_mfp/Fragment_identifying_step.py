import itertools
from rdkit import Chem
from rdkit.Chem import AllChem

class FragmentIdentifying:

    def identify_Fragment_Smarts(self, All_Scaffold_FilePath, qMol_Smarts, Final_all_Fragment_Dic):

        Final_qMol_Fragment_List = []
        S_all_Smarts = []
        S_all_classes = []
        f = open(All_Scaffold_FilePath, "r")

        while True:
            line = f.readline()
            if not line:break
            new_line = line.split("\t")
            S_all_classes.append(new_line[0])
            S_all_Smarts.append(new_line[1])

            qMol = Chem.MolFromSmarts(qMol_Smarts)

        for ai in range (0, len(S_all_Smarts)):

            if qMol.HasSubstructMatch(Chem.MolFromSmarts(S_all_Smarts[ai])):
                rm = AllChem.DeleteSubstructs(qMol, Chem.MolFromSmarts(S_all_Smarts[ai]))
                temp_query_mapping = list(qMol.GetSubstructMatch(Chem.MolFromSmarts(S_all_Smarts[ai])))

                Query_mapping=[]
                rm_idx=[]

                for atom_idx in temp_query_mapping:
                    atom_symbol = qMol.GetAtoms()[atom_idx].GetSymbol()
                    if atom_symbol == 'O' or atom_symbol == 'N':
                        try:
                            if qMol.GetAtoms()[atom_idx-1].GetSymbol() == 'C' and qMol.GetAtoms()[atom_idx+1].GetSymbol() == 'C':
                                rm_idx.append(atom_idx - 1)
                        except IndexError:
                            pass

                    else:
                        Query_mapping.append(atom_idx)

                mapping = Query_mapping + rm_idx

                Final_query_mapping = []
                for idx in mapping:
                    if mapping.count(idx) == 1:
                        Final_query_mapping.append(idx)

                Frags = str(Chem.MolToSmarts(rm)).split(".")

                Frag_Idx_List = []
                Frag_Idx_Smarts = []

                for bi in range(0, len(Frags)):
                    Molfrag = Chem.MolFromSmarts(Frags[bi])
                    try:
                        Frag_Idx_List.append(qMol.GetSubstructMatches(Molfrag))
                        Frag_Idx_Smarts.append(Frags[bi])
                    except RuntimeError:
                        Frag_Idx_List.append(())
                        Frag_Idx_Smarts.append(Frags[bi])

                Frag_Idx_List = list(itertools.chain(*Frag_Idx_List))
                Frag_Idx_List = list(set(Frag_Idx_List))

                temp_SFCP = []
                for ci in Frag_Idx_List:
                    temp_SFCP.append(ci[0]-1)

                SFCP = list(set(Final_query_mapping).intersection(temp_SFCP))

                for di in range(0, len(SFCP)):
                    if len(Frag_Idx_Smarts) > di and Frag_Idx_Smarts[di] in Final_all_Fragment_Dic:
                        coiso = Final_all_Fragment_Dic[Frag_Idx_Smarts[di]]
                        Final_qMol_Fragment_List.append(str(S_all_classes[ai])+"_"
                                                +"SFCP"+str(SFCP[di]+1)+"_"
                                                +str(coiso))
                    else:
                        Final_qMol_Fragment_List.append(str(S_all_classes[ai])+"_"
                                                   +"SFCP"+str(SFCP[di]+1))

        return Final_qMol_Fragment_List
