from web3_workers.faceswap import run_replicate_model_and_upload

import pytest

from web3_workers.dall_e_blender import blend_art_and_ingested_art, fetch_art


@pytest.mark.asyncio
async def test_faceswap():
    art_id = "c742007a-f993-43cf-949c-f6438ebfea7d"
    source_image = await fetch_art(art_id)
    source_image = source_image.get('img_picture')
    blended_image = "https://pixelpact.s3.us-east-2.amazonaws.com/generations/FS/01203f2a-648e-4909-adaa-a1c94dce28c3-FS-18271abb.png"
    generated_image = run_replicate_model_and_upload(source_image, blended_image)

    assert generated_image


if __name__ == '__main__':
    pytest.main()
