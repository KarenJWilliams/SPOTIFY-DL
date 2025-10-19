import React, { useState } from 'react';
import {
  Button,
  TextField,
  Container,
  Grid,
  Paper,
  Typography,
  Slider,
} from '@mui/material';

const SpotifyPredictor = () => {
  const [features, setFeatures] = useState({
    danceability: 0.5,
    energy: 0.5,
    loudness: -10,
    speechiness: 0.5,
    acousticness: 0.5,
    instrumentalness: 0.5,
    liveness: 0.5,
    valence: 0.5,
    tempo: 120,
  });
  const [prediction, setPrediction] = useState(null);

  const handlePredict = async () => {
    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(features),
      });
      const data = await response.json();
      setPrediction(data);
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <Container maxWidth="md">
      <Typography variant="h4" gutterBottom>
        Spotify Song Popularity Predictor
      </Typography>
      <Paper elevation={3} style={{ padding: 20, marginTop: 20 }}>
        <Grid container spacing={3}>
          {Object.entries(features).map(([feature, value]) => (
            <Grid item xs={12} key={feature}>
              <Typography gutterBottom>
                {feature.charAt(0).toUpperCase() + feature.slice(1)}
              </Typography>
              <Slider
                value={value}
                onChange={(e, newValue) =>
                  setFeatures({ ...features, [feature]: newValue })
                }
                min={feature === 'loudness' ? -60 : 0}
                max={feature === 'tempo' ? 250 : feature === 'loudness' ? 0 : 1}
                step={0.01}
                marks
                valueLabelDisplay="auto"
              />
            </Grid>
          ))}
        </Grid>
        <Button
          variant="contained"
          color="primary"
          onClick={handlePredict}
          style={{ marginTop: 20 }}
        >
          Predict Popularity
        </Button>
        {prediction && (
          <Typography variant="h6" style={{ marginTop: 20 }}>
            Predicted Popularity: {prediction.popularity_prediction.toFixed(2)}
          </Typography>
        )}
      </Paper>
    </Container>
  );
};

export default SpotifyPredictor;