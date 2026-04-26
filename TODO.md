# goal:
 * I want as many methods applied to this problem as possible 
   * I want pretty pictures 
     * I would like to show data analysis that represents the spread of variables and their distributions
     * I would like to show an understanding of the classical methods, their evaluation methods, and the models that dictate them
     * I would like to show an implementation of the modern methods including training curves 
     * I would like to show physically interpretable methods and how hybrid approaches can combine both worlds
     * I would like to show a complete comparison of performance across all models -> and therefore need to give each approach a standardized comparison metric (like isolating 20% of the data from the start)
   * then I want to be able to make a couple pareto fronts (idk if this even makes sense):
     * pareto front #1 -> I want to plot predictive R^2 over compute time/effort. I think the classical models will blow the AI models out of the park
     * pareto front #2 -> I want to plot predictive R^2 over the physical interpretability. maybe this means isolating a separate part of the datasets that  
     * pareto front #3 -> I want to plot the R^2 over development time; the machine learning models with more complex setup like physical models and complex systems should be penalized for that, while simple models (like linear regression) should be rewarded for their simplicity
   * what other figures would make for a pretty report that demonstrate an understanding of the ML processes being taken

# features:
 * inverse of tip deflection slope in units N/m (like a spring constant for the beam)
 * all the strain gauges are kinda useless and give the same data... so how many of them do you actually need -> this could be an expanded research question
 * remove n_steps as a training varaible -> its an artifact of the sim export and nothing more (it wouldnt be available to the pilot)
 * I want to explore more strain gauge simplification -- what if you take the xth percentile of the strain gauges as a feature, is that useful? what about a dual axis plot showing the strain gauge percentile chart overlaid with their R^2 to g_limit
   * long shot -> if you take all of the strain gauges and plot their percentiles, then fit a polynomial (low, ~3rd order) to their shape, then re-predict your maximum then you end up able to extract the max stress in a way that still provides smoothing and physical interpretability with increased robustness to extreme outliers. 
   * or just take the 95th percentile or sm
   * I would like a dual-axis plot that shows: 
     * a family of box plots for the percentiles of each strain value with the fitted polynomial
     * the R^2 predictive value of:
       * the discrete "always take the nth" max/2nd/3rd used to predict the max g loading
       * the continuous samples taken from the fitted polynomial R^2 to continuously interpolate non-whole percentiles (and smoothed!) 
 * what other features can I find from this dataset, and once I have a TON of them, how can I start sifting through to find the ones that are 1) independent and 2) useful predictors 


# Feature analysis
 * there is a current POD r2 vs k that includes the 26 origional and 7 engineered
   * 1) can you make the vertical axis to the same scale so that it shows the comparison between the scales
   * 2) can you add a third plot that shows all of the features Origional and Engineered 
   * 3) same changes to the pod_variance chart

# methods:
## very classical
 * linear regression
 * polynomial regression
## classical
 * GPR
 * random forest
 * ridge lasso
## modern
 * feedforward NN
 * gradient boosting
## very modern
 * PINN
   * multiple physical models
 * deep learning


