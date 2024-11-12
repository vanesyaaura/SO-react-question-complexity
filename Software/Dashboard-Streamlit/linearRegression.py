import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button, ButtonGroup } from '@/components/ui/button';
import { Tabs, TabList, TabPanels, Tab, TabPanel } from '@/components/ui/tabs';
import { Heading, Text } from '@/components/ui/typography';
import { LineChart, BarChart, PieChart } from 'recharts';

const StreamlitDashboard = ({ data }) => {
  // Sample data
  const sampleData = data;

  // Visualizations
  const LineChartComponent = () => {
    return (
      <LineChart width={600} height={400} data={sampleData}>
        <Line type="monotone" dataKey="Reputation" stroke="#8884d8" />
        <Line type="monotone" dataKey="pd_score" stroke="#82ca9d" />
        <Text>Reputation vs pd_score</Text>
      </LineChart>
    );
  };

  const BarChartComponent = () => {
    return (
      <BarChart width={600} height={400} data={sampleData}>
        <Bar dataKey="Reputation" fill="#8884d8" />
        <Bar dataKey="pd_score" fill="#82ca9d" />
        <Text>Reputation vs pd_score</Text>
      </BarChart>
    );
  };

  const PieChartComponent = () => {
    return (
      <PieChart width={600} height={400}>
        <Pie
          data={sampleData}
          dataKey="pd_score"
          nameKey="Reputation"
          cx="50%"
          cy="50%"
          outerRadius={100}
          fill="#8884d8"
          label
        />
        <Text>pd_score Breakdown</Text>
      </PieChart>
    );
  };

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Reputation vs pd_score Analysis Dashboard</CardTitle>
      </CardHeader>
      <CardContent>
        <Heading size="lg" className="mb-4">
          Research Overview
        </Heading>
        <Text className="mb-6">
          React, one of the most popular JavaScript libraries, often generates a multitude of technical questions from developers. These questions are frequently posted on discussion platforms like Stack Overflow. This research aims to analyze the complexity of React-related questions posted on Stack Overflow and evaluate their difficulty levels. The primary focus is on comparing the characteristics of answered and unanswered questions by measuring complexity based on various parameters such as question structure, usage of technical terminology, proficiency in writing source code, and the context of the questions.
        </Text>

        <Tabs>
          <TabList>
            <Tab>Data Overview</Tab>
            <Tab>Visualizations</Tab>
            <Tab>Machine Learning</Tab>
          </TabList>
          <TabPanels>
            <TabPanel>
              <Heading size="md" className="mb-4">Dataset Overview</Heading>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Heading size="sm" className="mb-2">Dataset Shape</Heading>
                  <Text>Rows: {sampleData.length} | Columns: {Object.keys(sampleData[0]).length}</Text>
                </div>
                <div>
                  <Heading size="sm" className="mb-2">Columns in Dataset</Heading>
                  <Text>{Object.keys(sampleData[0]).join(', ')}</Text>
                </div>
              </div>
              <Heading size="sm" className="mt-4 mb-2">Sample Data</Heading>
              <div className="overflow-x-auto">
                <table className="w-full border-collapse">
                  <thead>
                    <tr>
                      <th className="p-2 border text-left">Reputation</th>
                      <th className="p-2 border text-left">pd_score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sampleData.slice(0, 5).map((row, index) => (
                      <tr key={index}>
                        <td className="p-2 border">{row.Reputation}</td>
                        <td className="p-2 border">{row.pd_score}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </TabPanel>
            <TabPanel>
              <Heading size="md" className="mb-4">Data Visualizations</Heading>
              <ButtonGroup className="mb-4">
                <Button>Line Chart</Button>
                <Button>Bar Chart</Button>
                <Button>Pie Chart</Button>
              </ButtonGroup>
              <LineChartComponent />
              <BarChartComponent />
              <PieChartComponent />
            </TabPanel>
            <TabPanel>
              <Heading size="md" className="mb-4">Machine Learning Model</Heading>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Heading size="sm" className="mb-2">Reputation vs pd_score (Training Set)</Heading>
                  <LineChartComponent />
                </div>
                <div>
                  <Heading size="sm" className="mb-2">Reputation vs pd_score (Test Set)</Heading>
                  <LineChartComponent />
                </div>
              </div>
              <Heading size="sm" className="mt-4 mb-2">Model Coefficients and Intercept</Heading>
              <Text>Coefficient: 0.1234</Text>
              <Text>Intercept: 0.5678</Text>
              <Heading size="sm" className="mt-4 mb-2">Model Performance Metrics</Heading>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Text>Training Set MSE: 0.1234</Text>
                  <Text>Training Set R-squared: 0.5678</Text>
                </div>
                <div>
                  <Text>Test Set MSE: 0.1234</Text>
                  <Text>Test Set R-squared: 0.5678</Text>
                </div>
              </div>
              <Heading size="sm" className="mt-4 mb-2">Residuals Plot (Test Set)</Heading>
              <LineChartComponent />
            </TabPanel>
          </TabPanels>
        </Tabs>
      </CardContent>
    </Card>
  );
};

export default StreamlitDashboard;
