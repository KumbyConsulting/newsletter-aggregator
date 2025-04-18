# Kumby Consulting Newsletter Frontend

This is the frontend application for the Kumby Consulting Newsletter Aggregator, built with Next.js and React.

## Getting Started

First, set up your environment:

```bash
# Copy the example environment file
cp .env.example .env

# Install dependencies
npm install
```

Then, run the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Connecting to the Backend

By default, the frontend connects to a Flask backend running at `http://localhost:5000`. You can configure this in two ways:

1. Edit the `.env` file and change the `API_URL` value:
```
API_URL=http://your-backend-url
```

2. Or set an environment variable when starting the app:
```bash
API_URL=http://your-backend-url npm run dev
```

The application proxies all requests to `/api/*` endpoints to the configured backend URL.

## Features

- Real-time update status monitoring
- Article filtering and search
- Topic distribution visualization
- AI-powered insights
- Mobile-responsive design

## Building for Production

To create an optimized production build:

```bash
npm run build
```

Then start the production server:

```bash
npm start
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_URL` | Backend API URL | `http://localhost:5000` |
| `NEXT_PUBLIC_ENABLE_INSIGHTS` | Enable AI insights feature | `true` |

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
