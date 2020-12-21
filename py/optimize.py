import numpy as np
import pandas
from scipy.optimize import differential_evolution, OptimizeResult
from uk.py.loss_functions import loss_metrics, extract
from uk.py.calc_ubi import get_data, get_adult_amount


def optimize(input_dict, loss_metric, reform, verbose=True, **kwargs):


  '''
  Arguments:
  - input_dict = a dict with format {category: (min, max)} specifying the bounds for UBI amounts
                 for each category. If min == max, the amount is fixed.
  - loss_metric = the type of loss metric to be used in optimization.
  - reform = the type of reform to apply
  - verbose = bool specifying whether or not to print each function evaluation
  - **kwargs = kwargs for differential_evolution

  Returns:
  - return an OptimizeResult with the optimal solution.
  - prints and output_dict with the optimal solution.
  - prints the loss metrics for the optimal solution set.
  '''
  
  # Declare categories 
  CATEGORIES = ['senior', 'child', 'dis_1', 'dis_2', 'dis_3','NORTH_EAST', 'NORTH_WEST', 
                'YORKSHIRE', 'EAST_MIDLANDS', 'WEST_MIDLANDS', 'EAST_OF_ENGLAND', 'LONDON', 
                'SOUTH_EAST', 'SOUTH_WEST', 'WALES', 'SCOTLAND', 'NORTHERN_IRELAND']

  # Set bounds according to chosen reform.
  if reform == 'reform_1':
    bounds = [input_dict[i] for i in CATEGORIES[:2]]
    bounds += [(0,0)] * 14
  elif reform == 'reform_2':
    bounds = [input_dict[i] for i in CATEGORIES[:5]]
    bounds += [(0,0)] * 11
  elif reform == 'reform_3':
    bounds = [input_dict[i] for i in CATEGORIES]
  

  baseline_df, reform_base_df, budget = get_data()


  # Take the average value of each tuple to create array of starting values
  x = [((i[0] + i[1])/2) for i in bounds]
  
  # Add in the adult amount key
  CATEGORIES = ['adult'] + CATEGORIES

  def loss_func(x, args=(loss_metric)):
    loss_metric = args

    # Calculate loss with current solution (given the adult amount)
    loss = loss_metrics(x, baseline_df, reform_base_df, budget)[loss_metric]

    if verbose:
      senior, child, dis_1, dis_2, dis_3, regions = extract(x)
      adult_amount = get_adult_amount(reform_base_df, budget, senior, child,
                                      dis_1, dis_2, dis_3, regions, 
                                      individual=True)
      x = np.insert(x, 0, adult_amount)
      output_dict = {CATEGORIES[i]: x[i] for i in range(len(x))}

      # Print loss and corresponding solution set
      print ('Loss: {}'.format(loss))
      print (output_dict)

    return loss
  
  result = differential_evolution(func=loss_func, bounds=bounds, **kwargs)
  
  # Performance of optimal solutions on other loss metrics
  loss_dict = loss_metrics(result.x, baseline_df, reform_base_df, budget).to_dict()
  print ('Loss by all metrics:\n', loss_dict, '\n')

  # Get adult amount
  senior, child, dis_1, dis_2, dis_3, regions = extract(result.x)
  adult_amount = get_adult_amount(reform_base_df, budget, senior, child, dis_1,
                                  dis_2, dis_3, regions, individual=True)
  
  # Insert adult amount into optimal solution set
  result.x = np.insert(result.x, 0, adult_amount)

  # Print optimal loss
  print ('Optimal {}:'.format(loss_metric), result.fun, '\n')

  # Print optimal solution output_dict
  output_dict = {CATEGORIES[i]: result.x[i] for i in range(len(result.x))}
  print ('Optimal solution:\n', output_dict)

  return result
