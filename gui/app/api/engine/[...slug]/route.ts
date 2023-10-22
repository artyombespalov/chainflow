import { NextRequest, NextResponse } from "next/server";

const handler = async (req: Request, { params }: any) => {
  const { searchParams } = new URL(req.url);
  const requestUrl = params?.slug?.join("/") || "";

  const query = Object.entries(Object.fromEntries(searchParams))
    .map(([key, value]) => `${key}=${value}`)
    .join("&");

  const url = `https://chainflow-engine.dexguru.biz/engine-rest/${requestUrl}${
    query ? `?${query}` : ""
  }`;

  try {
    const result = await fetch(url, {
      headers: {
        "Content-Type": "application/json",
      },
      method: req.method,
      body: ["POST", "PUT"].includes(req.method)
        ? JSON.stringify(await req.json())
        : undefined,
    });

    if (!result.ok) {
      return NextResponse.json(
        {
          url: requestUrl,
          message: result.statusText,
          status: result.status,
          body: await result.json(),
        },
        {
          status: result.status,
        }
      );
    }

    return NextResponse.json(await result.json(), { status: 200 });
  } catch (e: any) {
    return NextResponse.json(
      { message: e.message, status: 500, description: "Server request failed" },
      { status: 500 }
    );
  }
};

export const GET = handler;
export const POST = handler;
export const PUT = handler;
