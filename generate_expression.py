from absl import app
from absl import flags

from render_images import create_image_from_expression
from td.environments import environments
from td.samplers import ConstrainedRandomSampler
from td.samplers.mutator import random_mutation, find_path

flags.DEFINE_string("environment", "tinysvgoffset", "Environment to evaluate.")
FLAGS = flags.FLAGS


def main(argv):
    env = environments[FLAGS.environment]()
    sampler = ConstrainedRandomSampler(env.grammar)
    expression = sampler.sample(env.grammar.start_symbol, 4, 4)
    print("Random expression:", expression)
    _ = create_image_from_expression(expression, env, output_path='test_render', filename='random_image')
    
    print("Mutating!")

    current_expression = expression
    for i in range(4):
        m = random_mutation(current_expression, env.grammar, sampler)
        # print(m.pretty(current_expression))
        current_expression = m.apply(current_expression)

    _ = create_image_from_expression(current_expression, env, output_path='test_render', filename='mutated_image')
    print(current_expression)


if __name__ == "__main__":
    app.run(main)