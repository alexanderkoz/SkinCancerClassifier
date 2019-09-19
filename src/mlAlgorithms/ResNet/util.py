def decode_predictions(preds, top=3):
    """Decodes the prediction of an ImageNet model.

    # Arguments
        preds: Numpy tensor encoding a batch of predictions.
        top: Integer, how many top-guesses to return.

    # Returns
        A list of lists of top class prediction tuples
        `(class_name, class_description, score)`.
        One list of tuples per sample in batch input.

    # Raises
        ValueError: In case of invalid shape of the `pred` array
            (must be 2D).
    """
    
    CLASS_INDEX = {
        '0': 'akiec', 
        '1': 'bcc', 
        '2': 'bkl', 
        '3': 'df', 
        '4': 'mel', 
        '5': 'nv', 
        '6': 'vasc'
    }

    if top > 7:
        top = 7

    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results