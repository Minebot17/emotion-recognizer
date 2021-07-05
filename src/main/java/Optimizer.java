import org.deeplearning4j.arbiter.MultiLayerSpace;
import org.deeplearning4j.arbiter.conf.updater.AdamSpace;
import org.deeplearning4j.arbiter.conf.updater.NesterovsSpace;
import org.deeplearning4j.arbiter.layers.LSTMLayerSpace;
import org.deeplearning4j.arbiter.layers.RnnOutputLayerSpace;
import org.deeplearning4j.arbiter.optimize.api.OptimizationResult;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultReference;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxTimeCondition;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.optimize.generator.RandomSearchGenerator;
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.integer.IntegerParameterSpace;
import org.deeplearning4j.arbiter.optimize.runner.LocalOptimizationRunner;
import org.deeplearning4j.arbiter.saver.local.FileModelSaver;
import org.deeplearning4j.arbiter.scoring.impl.EvaluationScoreFunction;
import org.deeplearning4j.arbiter.task.MultiLayerNetworkTaskCreator;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class Optimizer {
    public static void main(String[] args) throws IOException, InterruptedException {
        ContinuousParameterSpace learningRateHyperparam  = new ContinuousParameterSpace(0.00001, 0.01);
        IntegerParameterSpace layerSizeHyperparam  = new IntegerParameterSpace(512,4096);

        MultiLayerSpace hyperparameterSpace  = new MultiLayerSpace.Builder()
                .weightInit(WeightInit.XAVIER)
                .seed(System.currentTimeMillis())
                .updater(new AdamSpace(learningRateHyperparam))
                .addLayer( new LSTMLayerSpace.Builder()
                        .nIn(256)
                        .activation(Activation.TANH)
                        .nOut(layerSizeHyperparam)
                        .build())
                .addLayer( new RnnOutputLayerSpace.Builder()
                        .nOut(3)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(LossFunctions.LossFunction.MCXENT)
                        .build())
                .build();

        RandomSearchGenerator candidateGenerator = new RandomSearchGenerator(hyperparameterSpace, null);
        EvaluationScoreFunction scoreFunction = new EvaluationScoreFunction(Evaluation.Metric.RECALL);
        MaxTimeCondition terminationConditions = new MaxTimeCondition(60, TimeUnit.MINUTES);

        String baseSaveDirectory = "C:\\Users\\serpi\\Desktop\\repos\\EEGSetParser\\run\\arbiter";
        FileModelSaver modelSaver = new FileModelSaver(baseSaveDirectory);

        OptimizationConfiguration configuration = new OptimizationConfiguration.Builder()
                .candidateGenerator(candidateGenerator)
                .dataSource(EEGDataSource.class, null)
                .modelSaver(modelSaver)
                .scoreFunction(scoreFunction)
                .terminationConditions(terminationConditions)
                .build();

        LocalOptimizationRunner runner = new LocalOptimizationRunner(configuration, new MultiLayerNetworkTaskCreator());

        //Start the hyperparameter optimization
        runner.execute();

        String s = "Best score: " + runner.bestScore() + "\n" + "Index of model with best score: " + runner.bestScoreCandidateIndex() + "\n" + "Number of configurations evaluated: " + runner.numCandidatesCompleted() + "\n";
        System.out.println(s);


        //Get all results, and print out details of the best result:
        int indexOfBestResult = runner.bestScoreCandidateIndex();
        List<ResultReference> allResults = runner.getResults();

        OptimizationResult bestResult = allResults.get(indexOfBestResult).getResult();
        //val bestModel = bestResult.mode.asInstanceOf[MultiLayerNetwork];

        System.out.println("\n\nConfiguration of best model:\n");
        System.out.println(bestResult.toString());
        //System.out.println(bestModel.getLayerWiseConfigurations().toJson());
    }
}
