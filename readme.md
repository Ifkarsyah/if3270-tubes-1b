# IF3270 Tugas Besar 1 Bagian B: Implementasi Decision Tree Learning

## Implementasi modul myID

```
  Create a root node for the tree
  If all examples are positive, Return the single-node tree Root, with label = +.
  If all examples are negative, Return the single-node tree Root, with label = -.
  If number of predicting attributes is empty, then Return the single node tree Root,
  with label = most common value of the target attribute in the examples.
  Otherwise Begin
      A <- The Attribute that best classifies examples.
      Decision Tree attribute for Root = A.
      For each possible value, v_i, of A,
          Add a new tree branch below Root, corresponding to the test A = v_i.
          Let Examples(v_i) be the subset of examples that have the value v_i for A
          If Examples(v_i) is empty
              Then below this new branch add a leaf node with label = most common target value in the examples
          Else below this new branch add the subtree ID3 (Examples(v_i), Target_Attribute, Attributes – {A})
  End
  Return Root
```

## Implementasi myC4.5

- Overfitting training data dengan post pruning.
- Continuous-valued attribute: information gain dari kandidate.
- Alternative measures for selecting attributes: gain ratio.
- Handling missing attribute value: most common target value.

## How To Use

1. pip install -r requirement.txt
2. Pada file main.py ubah nama file "your_data_here.csv" menjadi data file yang ingin dilatih.
3. Fungsi ID3 akan otomasis memisahkan data menjadi trainging dan pruning, menghandle missing value, continuous-value, dan melakukan gain ration, serta terakhir melakukan pruning
