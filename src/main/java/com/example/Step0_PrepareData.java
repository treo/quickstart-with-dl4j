package com.example;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class Step0_PrepareData {
    public static void main(String... args) throws IOException {
        final String inputPathname = "X:/Churn_Modelling/Churn_Modelling.csv";
        final String outputTrainPathname = "X:/Churn_Modelling/Train/";
        final String outputTestPathname = "X:/Churn_Modelling/Test/";

        final double splitAt = 0.8;

        final File inputFile = new File(inputPathname);
        final List<String> lines = Files.readAllLines(inputFile.toPath());
        final String header = lines.get(0);
        lines.remove(0);

        final int splitPosition = (int)Math.round(splitAt * lines.size());


        final Random random = new Random();
        random.setSeed(0xC0FFEE);

        Collections.shuffle(lines, random);

        final ArrayList<String> train = new ArrayList<>(lines.subList(0, splitPosition));
        final ArrayList<String> test = new ArrayList<>(lines.subList(splitPosition, lines.size()));


        int i = 0;
        for (String line : train) {
            Files.write(Paths.get(outputTrainPathname+(i++)+".csv"), line.getBytes());
        }

        for (String line : test) {
            Files.write(Paths.get(outputTestPathname+(i++)+".csv"), line.getBytes());
        }
    }
}
