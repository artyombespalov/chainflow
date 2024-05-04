import pytest

from web3_workers.dall_e_blender import blend_art_and_ingested_art


@pytest.mark.asyncio
async def test_blend_art_and_ingested_art():
    art_id = "0500467d-29cf-453f-b2d9-85293f7bb5b7"
    ingested_art_id = "636728c9-753d-446d-b47c-1d50957ea5d5"
    generated_image = await blend_art_and_ingested_art(art_id, ingested_art_id)

    assert generated_image


if __name__ == '__main__':
    pytest.main()
