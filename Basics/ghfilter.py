import matplotlib as plt

weights = [158.0, 164.2, 160.3, 159.9, 162.1, 164.6,
           169.6, 167.4, 166.4, 171.0, 171.2, 172.6]

time_step = 1.0  # day
scale_factor = 4.0/10


def predict_using_gain_guess(estimated_weight, gain_rate, do_print=False):
    # storage for the filtered results
    estimates, predictions = [estimated_weight], []

    for z in weights:
        # predict new position
        predicted_weight = estimated_weight + gain_rate*time_step
        # update filter
        estimated_weight = predicted_weight + \
            scale_factor * (z-predicted_weight)
        #save and log
        estimates.append(estimated_weight)
        predictions.append(predicted_weight)
        if do_print:
            print("previous estimate: %8.2f, prediction: %8.2f, estimate: %8.2f" % (
                estimated_weight, predicted_weight, estimated_weight))
            print()
    return estimates, predictions


initial_estimate = 160
estimates, predictions = predict_using_gain_guess(
    estimated_weight=initial_estimate, gain_rate=1, do_print=True)
