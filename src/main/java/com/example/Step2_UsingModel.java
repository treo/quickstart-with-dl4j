package com.example;

import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.analysis.DataAnalysis;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.normalize.Normalize;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.parallelism.ParallelInference;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.util.Arrays;
import java.util.List;

public class Step2_UsingModel {
    public static void main(String... args) throws Exception {
        File modelSave = new File("X:/Churn_Modelling/model.bin");
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelSave);
        DataAnalysis analysis = DataAnalysis.fromJson(ModelSerializer.getObjectFromFile(modelSave, "dataanalysis"));
        Schema targetSchema = Schema.fromJson(ModelSerializer.getObjectFromFile(modelSave, "schema"));
        Schema schema = new Schema.Builder()
                .addColumnsInteger("Age", "Tenure", "Num Of Products", "Credit Score")
                .addColumnsDouble("Balance", "Estimated Salary")
                .addColumnCategorical("Geography", "France", "Germany", "Spain")
                .addColumnCategorical("Gender", "Female", "Male")
                .addColumnCategorical("Has Credit Card", "0", "1")
                .addColumnCategorical("Is Active Member", "0", "1")
                .build();


        String[] newOrder = targetSchema.getColumnNames().stream().filter(it -> !it.equals("Exited")).toArray(String[]::new);
        TransformProcess transformProcess = new TransformProcess.Builder(schema)
                .categoricalToOneHot("Geography", "Gender", "Has Credit Card", "Is Active Member")
                .integerToOneHot("Num Of Products", 1, 4)
                .normalize("Tenure", Normalize.MinMax, analysis)
                .normalize("Age", Normalize.Standardize, analysis)
                .normalize("Credit Score", Normalize.Log2Mean, analysis)
                .normalize("Balance", Normalize.Log2MeanExcludingMin, analysis)
                .normalize("Estimated Salary", Normalize.Log2MeanExcludingMin, analysis)
                .reorderColumns(newOrder)
                .build();


        List<Writable> record = RecordConverter.toRecord(schema, Arrays.asList(26, 8, 1, 547, 97460.1, 43093.67, "France", "Male", "1", "1"));
        List<Writable> transformed = transformProcess.execute(record);
        INDArray data = RecordConverter.toArray(transformed);

        int labelIndex = model.predict(data)[0];
        INDArray output = model.output(data, false);
        double[] result = model.output(data, false).toDoubleVector();

        System.out.println(labelIndex);
        System.out.println(output);
        System.out.println(Arrays.toString(result));

        ParallelInference wrapped = new ParallelInference.Builder(model).build();
        INDArray parOutput = wrapped.output(data);

        System.out.println(parOutput);
    }
}
