import numpy as np

# global config
Y_MAX = 7

def get_constant_dict(model_name):
    template_start_x = '\\%sstartx'
    template_node_distance = '\\%snodedistance'
    template_space_between = '\\%sspacebetween'
    template_layer_height = '\\%slayerheight'
    template_net_width = '\\%snetwidth'

    return {
            'start_x': template_start_x % model_name,
            'node_distance': template_node_distance % model_name,
            'space_between': template_space_between % model_name,
            'layer_height': template_layer_height % model_name,
            'net_width': template_net_width % model_name
            }


def texify_number(num):
    "uses a-j instead of 0-9 to work with tex"
    ret = ''

    if num == 0:
        return 'a'
    
    #  for i in range(num_digits):
    while num > 0:
        int_val = ord('a') + (num % 10)
        ret += chr(int_val)
        num //= 10
        
        #  if num == 0:
            #  ret += 'a'

    return ret[::-1]

def build_log_scale(smallest_exponent, largest_exponent):
    smallest_exponent = int(smallest_exponent)
    largest_exponent = int(largest_exponent)

    num_scales = largest_exponent - smallest_exponent + 1
    y_ticks = np.empty(num_scales*9+1)
    y_labels = []
    for i, exp in enumerate(range(smallest_exponent, largest_exponent+1)):
        base = 10**exp
        y_labels.append((base, "10^{%d}" % exp))
        for j in range(9):
            y_ticks[i*9+j] = (j+1)*base

    y_ticks[-1] = 10*base
    y_labels.append((10*base, "10^{%d}" % (exp + 1)))

    return y_ticks, y_labels

def calc_log_coord(lin_coord, y_min, y_max):
    scale_width = (y_max+0.05) - (y_min-0.05)
    log_coord = np.log10(lin_coord)
    return (log_coord-y_min)/scale_width

def generate_y_axis(y_min, y_max):
    y_ticks, y_labels = build_log_scale(y_min, y_max-1) # TODO: ugly workaround for scales
    ret = ''
    ret += '\\newcommand{\\ymax}{%f}\n' % Y_MAX
    ret += '\\newcommand{\\yticks}{'
    with np.errstate(all='raise'):
        try:
            for i,y in enumerate(y_ticks):
                coordinate = calc_log_coord(y, y_min, y_max)
            
                if i > 0:
                    ret += ','
                ret += '%f' % (coordinate*Y_MAX)
        except FloatingPointError as e:
            print("ERROR: %s" % e, file=sys.stderr)
            print("y: %f" % y, file=sys.stderr)

    ret += '}\n'

    # generate command for y labels
    label_command = "\\newcommand{\\ylabels}[1]{\n"
    for lin_y, label_text in y_labels:
        log_y = calc_log_coord(lin_y, y_min, y_max)

        current_label = "\\draw (#1, %f) node {\\tiny $%s$};\n" % (log_y * Y_MAX, label_text)
        label_command += current_label
    label_command += "}\n"

    ret += label_command

    return ret

