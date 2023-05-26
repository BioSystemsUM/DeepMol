import string

_SMILES_TOKENS = [
    'Al', 'As', 'B', 'Br', 'C', 'Cl', 'F', 'H', 'I', 'K', 'Li', 'N',
    'Na', 'O', 'P', 'S', 'Se', 'Si', 'Te',
    '(', ')', '[', ']', '=', '#', '%', '0', '1', '2', '3', '4', '5',
    '6', '7', '8', '9', '+', '-', 'se', 'te', 'c', 'n', 'o', 's']

_AVAILABLE_TOKENS = set(string.ascii_letters) - set(_SMILES_TOKENS)
