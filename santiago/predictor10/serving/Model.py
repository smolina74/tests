class Model():
    def predict(self,X,feature_names):
        # where X is an array of samples. Therefore this predict is responsible of obtaining 1 prediction per sample.
        # In this case X is coming as [{'id':<correlation_id>, 'values':[value1, value2, value3, ...]}]
        predictions_with_ids = []
        for sample in X:
            prediction_for_sample = {'id': sample['id'], 'values': [1,2,3]}
            predictions_with_ids.append(prediction_for_sample)
        print(predictions_with_ids)
        return predictions_with_ids