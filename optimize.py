from scipy.optimize import differential_evolution
from uk.py.loss_functions import loss_metrics

def optimize(input_dict, loss_metric):

  '''
  Arguments:
  - input_dict = a dict with format {category: (min, max)} specifying the bounds for UBI amounts
                 for each category. If min == max, the amount is fixed.
  - loss_metric = the type of loss metric to be used in optimization.

  Returns:
  - prints the loss and corresponding UBI amount per category solution set for each function evaluation.
  - return an OptimizeResult with the optimal solution.
  '''
  
  # Declare categories 
  CATEGORIES = ['senior', 'adult', 'child', 'dis_1', 'dis_2', 'dis_3','NORTH_EAST', 'NORTH_WEST', 
                'YORKSHIRE', 'EAST_MIDLANDS', 'WEST_MIDLANDS', 'EAST_OF_ENGLAND', 'LONDON', 
                'SOUTH_EAST', 'SOUTH_WEST', 'WALES', 'SCOTLAND', 'NORTHERN_IRELAND']

  bounds = [input_dict[i] for i in CATEGORIES]
  
  # Take the average value of each tuple to create array of starting values
  x = [((i[0] + i[1])/2) for i in bounds]

  def loss_func(x, args=(loss_metric)):
    loss_metric = args

    # Calculate loss with current solution.
    loss = loss_metrics(x)[loss_metric]

    # Print loss and corresponding solution set
    output_dict = {CATEGORIES[i]: x[i-1] for i in range(len(x))}
    print ('Loss: {}'.format(loss))
    print (output_dict)

    return loss
  
  result = differential_evolution(func=loss_func, bounds=bounds, maxiter=1)