from scipy.optimize import differential_evolution

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
  categories = ['senior', 'adult', 'child', 'dis_1', 'dis_2', 'dis_3','NORTH_EAST', 'NORTH_WEST', 
                'YORKSHIRE', 'EAST_MIDLANDS', 'WEST_MIDLANDS', 'EAST_OF_ENGLAND', 'LONDON', 
                'SOUTH_EAST', 'SOUTH_WEST', 'WALES', 'SCOTLAND', 'NORTHERN IRELAND']

  # Assign values for ordering
  senior = input_dict['senior']
  child = input_dict['child']
  dis_1 = input_dict['dis_1']
  dis_2 = input_dict['dis_2']
  dis_3 = input_dict['dis_3']
  NORTH_EAST = input_dict['NORTH_EAST']
  NORTH_WEST = input_dict['NORTH_WEST']
  YORKSHIRE = input_dict['YORKSHIRE']
  EAST_MIDLANDS = input_dict['EAST_MIDLANDS']
  WEST_MIDLANDS = input_dict['WEST_MIDLANDS']
  EAST_OF_ENGLAND = input_dict['EAST_OF_ENGLAND']
  LONDON = input_dict['LONDON']
  SOUTH_EAST = input_dict['SOUTH_EAST']
  SOUTH_WEST = input_dict['SOUTH_WEST']
  WALES = input_dict['WALES']
  SCOTLAND = input_dict['SCOTLAND']
  NORTHERN_IRELAND = input_dict['NORTHERN_IRELAND']


  # Declare bounds for each amount
  bounds = [senior, child, dis_1, dis_2, dis_3, NORTH_EAST, NORTH_WEST, YORKSHIRE, EAST_MIDLANDS,
            WEST_MIDLANDS, EAST_OF_ENGLAND, LONDON, SOUTH_EAST, SOUTH_WEST, WALES, SCOTLAND, NORTHER_IRELAND]
  
  # Take the average value of each tuple to create array of starting values

  x = [((i[0] + i[1])/2) for i in bounds]

  def loss_func(x, args=(loss_metric)):
    loss_metric = *args

    # Calculate loss with current solution.
    loss = loss_metrics(x)[loss_metric]

    # Print loss and corresponding solution set
    output_dict = {categories[i]: x[i-1] for i in range(len(x))
    print ('Loss: {}'.format(loss))
    print (output_dict)

    return loss
  
  result = differential_evolution(func=loss_func, bounds=bounds, maxiter=1)
