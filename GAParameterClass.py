import pygad
import numpy as np
import pandas as pd
from MP3NoiseEvalClass import MP3NoiseEvalClass
import argparse

class GAParameterClass:
    def inital_genespace(self):
        noise_low_boudary = 60
        noise_high_boudary = 90
        noise_step = 1
        clipping_low_boudary = 0
        clipping_high_boudary = 2.1
        clipping_step = 0.1
        limiter_low_boudary = 0
        limiter_high_boudary = 2.1
        limiter_step = 0.1


        v_noise_gene_space = {"low": noise_low_boudary, "high": noise_high_boudary, "step":noise_step}
        v_clipping_gene_space = {"low": clipping_low_boudary, "high": clipping_high_boudary, "step":clipping_step}
        v_limiterII_gene_space = {"low": limiter_low_boudary, "high": limiter_high_boudary, "step":limiter_step}

        d_noise_gene_space = {"low": noise_low_boudary, "high": noise_high_boudary, "step":noise_step}
        d_clipping_gene_space = {"low": clipping_low_boudary, "high": clipping_high_boudary, "step":clipping_step}
        d_limiterII_gene_space = {"low": limiter_low_boudary, "high": limiter_high_boudary, "step":limiter_step}

        b_noise_gene_space = {"low": noise_low_boudary, "high": noise_high_boudary, "step":noise_step}
        b_clipping_gene_space = {"low": clipping_low_boudary, "high": clipping_high_boudary, "step":clipping_step}
        b_limiterII_gene_space = {"low": limiter_low_boudary, "high": limiter_high_boudary, "step":limiter_step}

        o_noise_gene_space = {"low": noise_low_boudary, "high": noise_high_boudary, "step":noise_step}
        o_clipping_gene_space = {"low": clipping_low_boudary, "high": clipping_high_boudary, "step":clipping_step}
        o_limiterII_gene_space = {"low": limiter_low_boudary, "high": limiter_high_boudary, "step":limiter_step}


        self.gene_type = [int, [float, 1],[float, 1],int, [float, 1],[float, 1],int, [float, 1],[float, 1],int, [float, 1],[float, 1]]
        self.gene_space = [v_noise_gene_space,v_clipping_gene_space,v_limiterII_gene_space,
                    d_noise_gene_space,d_clipping_gene_space,d_limiterII_gene_space,
                    b_noise_gene_space,b_clipping_gene_space,b_limiterII_gene_space,
                    o_noise_gene_space,o_clipping_gene_space,o_limiterII_gene_space,]

    def run(self):
        self.GA_instance = pygad.GA(num_generations=self.num_generations,
                       num_parents_mating=self.num_parents_mating,
                       num_genes = self.num_genes,
                       on_generation=self.on_gen,
                       sol_per_pop= self.sol_per_pop,
                       fitness_func=self.peaq_func,
                       gene_type = self.gene_type,
                       gene_space = self.gene_space,
                       mutation_percent_genes=25,
                       keep_elitism = 1,
                       save_best_solutions=True,
                       parallel_processing=None)
        self.GA_instance.run()

    def OutputDateFrame(self):
        score_df = pd.DataFrame(self.GA_instance.best_solutions_fitness, columns=['score'])
        manip_df = pd.DataFrame(self.GA_instance.best_solutions, columns=['V_Noise_SNR', 'V_Clipping_Percentage', 'V_Limiter_Threshold','D_Noise_SNR', 'D_Clipping_Percentage', 'D_Limiter_Threshold','B_Noise_SNR', 'B_Clipping_Percentage', 'B_Limiter_Threshold','O_Noise_SNR', 'O_Clipping_Percentage', 'O_Limiter_Threshold'])
        full_df = pd.concat([score_df, manip_df], axis=1)
        return score_df, manip_df, full_df

    def on_gen(self,ga_instance):
        print("Generation : ", ga_instance.generations_completed)
        best_solutions = tuple(ga_instance.best_solutions[ga_instance.generations_completed])
        print(f"The last best Solution : ", {best_solutions})
        best_fitness = ga_instance.best_solutions_fitness[ga_instance.generations_completed-1]
        print(f"Fitness of the last best solution :", {best_fitness})

    def peaq_func(self,ga_instance, solution, solution_idx):
        v_int_noise = solution[0]
        v_float_clippingper = solution[1]
        v_float_IIdynamic = solution[2]
        d_int_noise = solution[3]
        d_float_clippingper = solution[4]
        d_float_IIdynamic = solution[5]
        b_int_noise = solution[6]
        b_float_clippingper = solution[7]
        b_float_IIdynamic = solution[8]
        o_int_noise = solution[9]
        o_float_clippingper = solution[10]
        o_float_IIdynamic = solution[11]    
        filename = f'audio_mixing_V_SNR_{v_int_noise}.0_CP_{v_float_clippingper}_IITH_{v_float_IIdynamic}_D_SNR_{d_int_noise}.0_CP_{d_float_clippingper}_IITH_{d_float_IIdynamic}_B_SNR_{b_int_noise}.0_CP_{b_float_clippingper}_IITH_{b_float_IIdynamic}_O_SNR_{o_int_noise}.0_CP_{o_float_clippingper}_IITH_{o_float_IIdynamic}.wav'
        Gener_Audio = self.Noise_Generator_MP3.TestNoisedFullTrack(solution,filename,isNormalised=False)
        #print(Gener_Audio)
        score = self.Noise_Generator_MP3.MeasurePEAQOutputsVsRef(Gener_Audio,64,self.Referece_File)
        #return float(score)
        return abs(round(float(score),2))

    def __init__(self, num_generations=10,mutation_percent_genes=25,num_parents_mating=2,sol_per_pop=5,crossover_type="uniform"):
        self.num_generations=num_generations
        self.num_genes=12
        self.num_parents_mating = num_parents_mating
        self.sol_per_pop = sol_per_pop
        self.crossover_type = crossover_type
        self.mutation_percent_genes = mutation_percent_genes
        ###Loading the algorithm
        self.Mixing_Path = '/home/codecrack/Jnotebook/44k1/Gospel'
        self.Noise_Generator_MP3 = MP3NoiseEvalClass(self.Mixing_Path,StartingTime=8)
        self.Referece_File = self.Noise_Generator_MP3.TestNoisedFullTrack([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"Reference.wav",isNormalised=False)
        self.inital_genespace()

