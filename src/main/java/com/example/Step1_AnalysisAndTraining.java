package com.example;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.normalize.Normalize;
import org.datavec.api.transform.ui.HtmlAnalysis;
import org.datavec.local.transforms.AnalyzeLocal;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.EvaluationAveraging;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;

import java.io.File;
import java.util.Random;

public class Step1_AnalysisAndTraining {
    public static void main(String... args) throws Exception {
        Random random = new Random();
        random.setSeed(0xC0FFEE);
        FileSplit inputSplit = new FileSplit(new File("X:/Churn_Modelling/Train/"), random);
        CSVRecordReader recordReader = new CSVRecordReader();
        recordReader.initialize(inputSplit);

        Schema schema = new Schema.Builder()
                .addColumnsInteger("Row Number", "Customer Id")
                .addColumnString("Surname")
                .addColumnInteger("Credit Score")
                .addColumnCategorical("Geography", "France", "Germany", "Spain")
                .addColumnCategorical("Gender", "Female", "Male")
                .addColumnsInteger("Age", "Tenure")
                .addColumnDouble("Balance")
                .addColumnInteger("Num Of Products")
                .addColumnCategorical("Has Credit Card", "0", "1")
                .addColumnCategorical("Is Active Member", "0", "1")
                .addColumnDouble("Estimated Salary")
                .addColumnCategorical("Exited", "0", "1")
                .build();

        DataAnalysis analysis = AnalyzeLocal.analyze(schema, recordReader);
        HtmlAnalysis.createHtmlAnalysisFile(analysis, new File("X:/Churn_Modelling/analysis.html"));

        TransformProcess transformProcess = new TransformProcess.Builder(schema)
                .removeColumns("Row Number", "Customer Id", "Surname")
                .categoricalToOneHot("Geography", "Gender", "Has Credit Card", "Is Active Member")
                .integerToOneHot("Num Of Products", 1, 4)
                .normalize("Tenure", Normalize.MinMax, analysis)
                .normalize("Age", Normalize.Standardize, analysis)
                .normalize("Credit Score", Normalize.Log2Mean, analysis)
                .normalize("Balance", Normalize.Log2MeanExcludingMin, analysis)
                .normalize("Estimated Salary", Normalize.Log2MeanExcludingMin, analysis)
                .build();

        Schema finalSchema = transformProcess.getFinalSchema();

        TransformProcessRecordReader trainRecordReader = new TransformProcessRecordReader(new CSVRecordReader(), transformProcess);
        trainRecordReader.initialize(inputSplit);

        int batchSize = 80;
        RecordReaderDataSetIterator trainIterator = new RecordReaderDataSetIterator.Builder(trainRecordReader, batchSize)
                .classification(finalSchema.getIndexOfColumn("Exited"), 2)
                .build();


        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(0xC0FFEE)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.TANH)
                .updater(new Adam.Builder().learningRate(0.15).build())
                .l2(0.00316)
                .list(
                        new DenseLayer.Builder().nOut(25).build(),
                        new DenseLayer.Builder().nOut(25).build(),
                        new DenseLayer.Builder().nOut(25).build(),
                        new DenseLayer.Builder().nOut(25).build(),
                        new DenseLayer.Builder().nOut(25).build(),
                        new OutputLayer.Builder(new LossMCXENT()).nOut(2).activation(Activation.SOFTMAX).build()
                )
                .setInputType(InputType.feedForward(finalSchema.numColumns() - 1))
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(config);
        model.init();

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);

        model.addListeners(new ScoreIterationListener(100));
        model.addListeners(new StatsListener(statsStorage, 250));

        model.fit(trainIterator, 100);

        TransformProcessRecordReader testRecordReader = new TransformProcessRecordReader(new CSVRecordReader(), transformProcess);
        testRecordReader.initialize( new FileSplit(new File("X:/Churn_Modelling/Test/")));
        RecordReaderDataSetIterator testIterator = new RecordReaderDataSetIterator.Builder(testRecordReader, batchSize)
                .classification(finalSchema.getIndexOfColumn("Exited"), 2)
                .build();

        Evaluation evaluate = model.evaluate(testIterator);
        System.out.println(evaluate.stats());
        System.out.println("MCC: "+evaluate.matthewsCorrelation(EvaluationAveraging.Macro));

        /*
========================Evaluation Metrics========================
 # of classes:    2
 Accuracy:        0,8715
 Precision:       0,8233
 Recall:          0,7127
 F1 Score:        0,5724
Precision, recall & F1: reported for positive class (class 1 - "1") only


=========================Confusion Matrix=========================
    0    1
-----------
 1571   54 | 0 = 0
  203  172 | 1 = 1

Confusion matrix format: Actual (rowClass) predicted as (columnClass) N times
==================================================================
MCC: 0.5244999138768601
         */

        File modelSave = new File("X:/Churn_Modelling/model.bin");
        model.save(modelSave);
        ModelSerializer.addObjectToFile(modelSave, "dataanalysis", analysis.toJson());
        ModelSerializer.addObjectToFile(modelSave, "schema", finalSchema.toJson());
    }
}
