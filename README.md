# DeepWeather

DeepWeather is a deep learning approach to improving weather forecasting accuracy by supplementing existing weather forecasts with a variety of satellite images.

The intuition behind DeepWeather is as follows: Since plants can influence the surrounding climate (e.g., through processes such as transpiration and evaporation), then using certain vegetation features as inputs to a neural network (given a numerical forecast as a prior) may result in more precise forecasts. 

It is worth mentioning that DeepWeather is still in its early stages. Continuous effort will be poured into improving the results and making the code more readable.

## Dataset

The dataset used in this project can be divided into two sections, corresponding to two data sources. For both data sources, data was collected for 500 random, global weather stations.

1) **OpenWeather** was used to collect both weather forecasts (to be used as a prior for weather predictions in this model) and observed weather data (to be used as ground-truth). Daily weather data was collected using OpenWeather's API for each of the 500 stations over 16 days. Data entries consisted of temperature (min, max, average) and humidity percentage for weather observations. The same features were used for weather forecasts, with the addition of cloud percentage.
2) **Sentinel Hub** was used to collect satellite images captured by the Sentinel-2 mission. Three types of images were collected for each station on each of the 16 days: True-color composite images (correspond to how humans perceive the Earth), EVI index images (correlated with vegetation greenness), and NDMI index images (correlated with crop moisture levels). All images were collected over an 8KM radius around the selected weather stations, and resized to 512 x 512 for computational efficiency.

## Implementation

DeepWeather was implemented in PyTorch and is available in this repo as a Jupyter Notebook. In what follows, implementation specifics will be discussed.

### Inputs

The inputs to our model consist of two classes:
1) Weather forecasts -> 1x5 vector of numerical data
2) Satellite images (stack of three satellite images (true-color, EVI, NDMI)) -> 512x512x9 input.

### Output

The model outputs a 1x4 vector corresponding to an improved weather forecast and is compared to the ground-truth weather observations (which are also 1x4) during training.

### Model

DeepWeather consists of a CNN which takes as input the stack of RGB satellite images (512x512x9), whose output is flattened into a 1D vector. Weather forecasts (1x5) are concatented to the flattened outputs of the CNN. Then, the concatenated 1D vector is finally fed into a 3-layer fully connected network which outputs a modified weather forecast.

The CNN’s input is a stack of the three images (True Color, EVI, NDMI), and the CNN consists of three semi-identical blocks. Each block consists of a 2D convolutional layer with a kernel size of 3 and a stride of 2, followed by a Leaky Rectified Linear Unit (Leaky ReLU), a 2D max-pooling layer with a kernel size of 2 and a stride of 2, a dropout layer with a dropout probability of 0.3, and finally a 2D batch normalization layer. The only difference between the blocks is that the number of output channels for each convolutional layer is 32, 64, and 128 respectively.

The three fully connected layers following the CNN’s output contain 1024, 128, and 4 neurons respectively. Each of the 4 neurons corresponds to one of the 4 weather conditions the model predicts (average temperature, minimum temperature, maximum temperature, humidity percentage).

The diagram below sums up the model architecture.

![Model Architecture](https://i.ibb.co/GWBRkXp/arch.png)
