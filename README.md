# AA2-Embeddings

Branch relativo ao trabalho prático da unidade curricular de Aprendizagem Automática 2, lecionada pelo Professor Doutor Miguel Rocha.

Seguido pela seguinte proposta de trabalho:

>Proposta A. 2 - Desenvolvimento de abordagens para a criação de embeddings moleculares baseados em algoritmos de NLP para integração numa ferramenta de machine learning existente (DeepMol).
>
>>Os resultados alcançados, nas últimas décadas, pelo uso de abordagens de inteligência artificial em áreas como visão computacional e processamento de linguagem natural levaram a um uso mais generalizado dessas abordagens noutras áreas. Um desses casos passa pela aplicação de aborgadens de Machine e Deep Learning em tópicos de quimioinformatica onde o objetivo passa pela identificação e design de moléculas com propriedades específicas. Para isto, o processo de implementação de métodos que sejam capazes de criar representações vetoriais que caracterizem substruturas moleculares assume grande importância. Hoje em dia existem múltiplas técnicas baseadas em modelos de NLP capazes de eficientemente criar embeddings moleculares a partir de representações como SMILES. Exemplos disso são os algoritmos Mol2vec, Seq2seq Fingerprint, SMILES Transformer, Message Passing Neural Networks for SMILES, SMILES-BERT entre muitos outros. Desta forma, o objetivo deste projeto passa pela implementação e integração de um módulo de abordagens baseadas em algoritmos de NLP para a criação de embeddings moleculares numa ferramenta de machine learning de classificação de compostos. Inicialmente, as tarefas passariam por explorar a pipeline existente usando um dataset como case study***. Depois de perceber a estrutura e organização da pipeline iria proceder-se à implementação e integração do módulo para criação de embeddings moleculares. O projeto será desenvolvido usando a linguagem Python

O grupo decidiu começar pela exploração dos seguintes algoritmos:

- Mol2Vec
- Seq2Seq2 Fingerprint
- SMILES Transformer

Nesta fase inicial, o algoritmo que melhores resultados obteve, numa óptica de implementação e de integração com o DeepMol, foi o SMILES Transformer.

