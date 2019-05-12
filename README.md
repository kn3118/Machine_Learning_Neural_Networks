### Neural Network Coursework

## Files
- nn_lib.py
- learn_FM.py
- learn_ROI.py
- Neural network project report

## How to run our predict_hidden for Part 2 and Part 3
Input either the entire dataset or just the X_input data (first three columns). We are expecting a text file. Replace "ROI_dataset.dat" with your own dataset
```
if __name__ == "__main__":

    dataset = np.loadtxt("ROI_dataset.dat")

    predictions = predict_hidden(dataset)
    print("Predictions:")
    print(predictions)

```
Once you include your dataset as above, run ```python3 learn_ROI.py``` or ```python3 learn_FM.py```