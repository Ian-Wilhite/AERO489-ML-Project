


features:
 * inverse of tip deflection slope in units N/m (like a spring constant for the beam)
 * all the strain gauges are kinda useless and give the same data... so how many of them do you actually need -> this could be an expanded research question
 * remove n_steps as a training varaible -> its an artifact of the sim export and nothing more (it wouldnt be available to the pilot)
 * I want to explore more strain gauge simplification -- what if you take the xth percentile of the strain gauges as a feature, is that useful? what about a dual axis plot showing the strain gauge percentile chart overlaid with their R^2 to g_limit

