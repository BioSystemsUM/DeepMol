from rdkit import Chem

class ScaffoldMatching:

    # Input Smarts
    def match_all_scaffold_smarts(self, FilePath, qMol_Smarts):

        Final_all_scaffold_Match_List = []
        S_all_Smarts = []
        S_all_classes = []
        f = open(FilePath, "r")

        while True:
            line = f.readline()
            if not line:break
            new_line = line.split("\t")
            S_all_classes.append(new_line[0])
            S_all_Smarts.append(new_line[1])

        for ai in range(0, len(S_all_Smarts)):
            qMol = Chem.MolFromSmarts(qMol_Smarts)
            if qMol.HasSubstructMatch(Chem.MolFromSmarts(S_all_Smarts[ai])):
                Final_all_scaffold_Match_List.append(S_all_classes[ai])


        return Final_all_scaffold_Match_List

    # Input Mol
    def match_All_Scaffold_Mol(self, FilePath, qMol):

        Final_all_scaffold_Match_List = []
        S_all_Smarts = []
        S_all_classes = []
        f = open(FilePath, "r")

        while True:
            line = f.readline()
            if not line:break
            new_line = line.split("\t")
            S_all_classes.append(new_line[0])
            S_all_Smarts.append(new_line[1])

        for ai in range(0, len(S_all_Smarts)):
            if qMol.HasSubstructMatch(Chem.MolFromSmarts(S_all_Smarts[ai])):
                Final_all_scaffold_Match_List.append(S_all_classes[ai])

        return Final_all_scaffold_Match_List


    # Input Smarts
    def match_Scaffold_Lv1_Smarts(self, FilePath, qMol_Smarts):

        Final_scaffold_Lv1_Match_List = []
        S_Lv1_Smarts = []
        S_Lv1_classes = []
        f = open(FilePath, "r")

        while True:
            line = f.readline()
            if not line:break
            new_line = line.split("\t")
            S_Lv1_classes.append(new_line[0])
            S_Lv1_Smarts.append(new_line[1])

        for ai in range(0, len(S_Lv1_Smarts)):
            qMol = Chem.MolFromSmarts(qMol_Smarts)
            if qMol.HasSubstructMatch(Chem.MolFromSmarts(S_Lv1_Smarts[ai])):
                Final_scaffold_Lv1_Match_List.append(S_Lv1_classes[ai])

        return Final_scaffold_Lv1_Match_List

    # Input Mol
    def match_Scaffold_Lv1_Mol(self, FilePath, qMol):

        Final_scaffold_Lv1_Match_List = []
        S_Lv1_Smarts = []
        S_Lv1_classes = []
        f = open(FilePath, "r")

        while True:
            line = f.readline()
            if not line:break
            new_line = line.split("\t")
            S_Lv1_classes.append(new_line[0])
            S_Lv1_Smarts.append(new_line[1])

        for ai in range(0, len(S_Lv1_Smarts)):
            if qMol.HasSubstructMatch(Chem.MolFromSmarts(S_Lv1_Smarts[ai])):
                Final_scaffold_Lv1_Match_List.append(S_Lv1_classes[ai])

        return Final_scaffold_Lv1_Match_List

    # Input Smarts
    def match_Scaffold_Lv2_Smarts(self, FilePath, qMol_Smarts):

        Final_scaffold_Lv2_Match_List = []
        S_Lv2_Smarts = []
        S_Lv2_classes = []
        f = open(FilePath, "r")

        while True:
            line = f.readline()
            if not line:break
            new_line = line.split("\t")
            S_Lv2_classes.append(new_line[0])
            S_Lv2_Smarts.append(new_line[1])

        for ai in range(0, len(S_Lv2_Smarts)):
            qMol = Chem.MolFromSmarts(qMol_Smarts)
            if qMol.HasSubstructMatch(Chem.MolFromSmarts(S_Lv2_Smarts[ai])):
                Final_scaffold_Lv2_Match_List.append(S_Lv2_classes[ai])

        return Final_scaffold_Lv2_Match_List

    # Input Mol
    def match_Scaffold_Lv2_Mol(self, FilePath, qMol):

        Final_scaffold_Lv2_Match_List = []
        S_Lv2_Smarts = []
        S_Lv2_classes = []
        f = open(FilePath, "r")

        while True:
            line = f.readline()
            if not line:break
            new_line = line.split("\t")
            S_Lv2_classes.append(new_line[0])
            S_Lv2_Smarts.append(new_line[1])

        for ai in range(0, len(S_Lv2_Smarts)):
            if qMol.HasSubstructMatch(Chem.MolFromSmarts(S_Lv2_Smarts[ai])):
                Final_scaffold_Lv2_Match_List.append(S_Lv2_classes[ai])

        return Final_scaffold_Lv2_Match_List

    def get_Scaffolds_all_classes(self, FilePath):

        Scaffold_all_classes = []

        f = open(FilePath, "r")

        while True:
            line = f.readline()
            if not line: break
            new_line = line.split("\t")
            Scaffold_all_classes.append(new_line[0])

        return Scaffold_all_classes

    def get_Scaffolds_all_Smarts(self, FilePath):

        Scaffold_all_Smarts = []

        f = open(FilePath, "r")
        while True:
            line = f.readline()
            if not line: break
            new_line = line.split("\t")
            Scaffold_all_Smarts.append(new_line[1])

        return Scaffold_all_Smarts

    def get_Scaffold_all_Dictionary(self, FilePath):

        Scaffold_all_Dic = {}
        Scaffold_all_classes = []
        Scaffold_all_Smarts = []

        f = open(FilePath, "r")

        while True:
            line = f.readline()
            if not line: break
            new_line = line.split("\t")
            Scaffold_all_classes.append(new_line[0])
            Scaffold_all_Smarts.append(new_line[1])

        for ai in range(0, len(Scaffold_all_classes)):
            Scaffold_all_Dic[Scaffold_all_Smarts[ai]] = Scaffold_all_classes[ai]

        return Scaffold_all_Dic

    def get_Scaffolds_Lv1_classes(self, FilePath):

        Scaffold_Lv1_classes = []

        f = open(FilePath, "r")

        while True:
            line = f.readline()
            if not line:break
            new_line = line.split("\t")
            Scaffold_Lv1_classes.append(new_line[0])

        return Scaffold_Lv1_classes

    def get_Scaffolds_Lv1_Smarts(self, FilePath):

        Scaffold_Lv1_Smarts = []

        f = open(FilePath, "r")
        while True:
            line = f.readline()
            if not line: break
            new_line = line.split("\t")
            Scaffold_Lv1_Smarts.append(new_line[1])

        return Scaffold_Lv1_Smarts

    def get_Scaffold_Lv1_Dictionary(self, FilePath):

        Scaffold_Lv1_Dic = {}
        Scaffold_Lv1_classes = []
        Scaffold_Lv1_Smarts = []

        f = open(FilePath, "r")

        while True:
            line = f.readline()
            if not line: break
            new_line = line.split("\t")
            Scaffold_Lv1_classes.append(new_line[0])
            Scaffold_Lv1_Smarts.append(new_line[1])

        for ai in range(0, len(Scaffold_Lv1_classes)):
            Scaffold_Lv1_Dic[Scaffold_Lv1_Smarts[ai]] = Scaffold_Lv1_classes[ai]

        return Scaffold_Lv1_Dic

    def get_Scaffolds_Lv2_classes(self, FilePath):

        Scaffold_Lv2_classes = []

        f = open(FilePath, "r")

        while True:
            line = f.readline()
            if not line:break
            new_line = line.split("\t")
            Scaffold_Lv2_classes.append(new_line[0])

        return Scaffold_Lv2_classes

    def get_Scaffolds_Lv2_Smarts(self, FilePath):

        Scaffold_Lv2_Smarts = []

        f = open(FilePath, "r")
        while True:
            line = f.readline()
            if not line: break
            new_line = line.split("\t")
            Scaffold_Lv2_Smarts.append(new_line[1])

        return Scaffold_Lv2_Smarts

    def get_Scaffold_Lv2_Dictionary(self, FilePath):

        Scaffold_Lv2_Dic = {}
        Scaffold_Lv2_classes = []
        Scaffold_Lv2_Smarts = []

        f = open(FilePath, "r")

        while True:
            line = f.readline()
            if not line: break
            new_line = line.split("\t")
            Scaffold_Lv2_classes.append(new_line[0])
            Scaffold_Lv2_Smarts.append(new_line[1])

        for ai in range(0, len(Scaffold_Lv2_classes)):
            Scaffold_Lv2_Dic[Scaffold_Lv2_Smarts[ai]] = Scaffold_Lv2_classes[ai]

        return Scaffold_Lv2_Dic





