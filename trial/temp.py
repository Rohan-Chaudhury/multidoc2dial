def predict_doc_at(query, k=1):
    """
    Predict which document is matched to the given query.

    :param query: input query
    :type query: str (or list of strs)
    :param k: number of returning docs
    :type k: int 
    :return: return the document name
    """
    query_embedding = get_embeddings(query)
    predictions, log_p_predictions = PLDA_classifier.predict(query_embedding,
    predictions = predictions[:k]
    sum_log = np.sum(np.exp(-log_p_predictions))
    accuracy = list(map(lambda x: np.exp(-x) / sum_log,
                        log_p_predictions[predictions]))
    predictions = list(map(lambda x: labels[x], predictions))
    return accuracy, predictions