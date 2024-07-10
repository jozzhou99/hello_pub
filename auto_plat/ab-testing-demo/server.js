const express = require('express');
const app = express();
const port = 3000;

app.use(express.static('public'));
app.use(express.urlencoded({ extended: true }));

app.set('view engine', 'ejs');
app.set('views', './views');

app.get('/', (req, res) => {
    res.render('index', { dataLoaded: false });
});

app.post('/load-data', (req, res) => {
    res.render('index', { dataLoaded: true });
});

app.post('/select-metric', (req, res) => {
    const metricType = req.body.metricType;
    res.render('step3', { metricType, validationPassed: false });
});

app.post('/validate-data', (req, res) => {
    const metricType = req.body.metricType;
    res.render('step3', { metricType, validationPassed: true });
});

app.post('/pre-test', (req, res) => {
    res.render('step4', {
        varianceBefore: { group1: 2.2313, group2: 1.9982 },
        varianceAfter: { group1: 0.812837, group2: 0.77671 }
    });
});

app.post('/run-test', (req, res) => {
    const { multipleTesting, fdr } = req.body;
    const pValue = 0.03;
    const significant = pValue < 0.05;
    res.render('result', {
        progress: 100,
        testStatistic: 2.54,
        pValue,
        multipleTesting: multipleTesting === 'on',
        fdr: fdr || 'Not specified',
        significant
    });
});

app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});
