
   Users can run the code along with the CMAPSS Data set 
    
  Objective:
        The project aims to implement deep learning-based predictive maintenance for predicting the Residual Useful Life (RUL) of turbofan degradation in aircraft engines.

  Dataset:
        C-MAPSS data sets from the Prognostics Centre of Excellence (PCoE) at NASA are used for training and testing the predictive maintenance model.

  Model Comparison:
        The project evaluates the performance of three different deep learning algorithms for time-series prediction: Long Short-Term Memory (LSTM), Bidirectional LSTM, and Gated Recurrent Units with LSTM (GRU-LSTM).

  Prescriptive Analysis:
        After predicting the RUL, the project goes a step further by calculating the prescription for the aircraft.
        The prescription involves estimating which part of the engine needs to be serviced first to significantly increase the RUL of the aircraft engine.

  Algorithm Performance:
        Performance analysis is conducted to compare the effectiveness of the three deep learning algorithms.
        Bidirectional LSTM (BiLSTM) outperforms the other algorithms in terms of MAE or other relevant metrics.

  Result Interpretation:
        The project provides insights into the performance of different deep learning architectures and their suitability for the predictive maintenance task.

  Practical Application:
        The results obtained from the predictive maintenance model can be practically applied to guide maintenance decisions for aircraft engines by choosing which part to service if failure does occur.
        The prescribed maintenance actions can potentially enhance the remaining useful life of the engines.

  Overall Conclusion:
        The project concludes by highlighting the significance of using deep learning for predictive maintenance in the context of aircraft engines and recommends the use of Bidirectional LSTM based on superior performance.

The Following project was accomplished by training multiple types of models, comparing their Mean Absolute Error and other metric, finding the mutual information so that we obtain the part which effects the engine the most for it future predictions give out the best combination of sensors to be fixed
