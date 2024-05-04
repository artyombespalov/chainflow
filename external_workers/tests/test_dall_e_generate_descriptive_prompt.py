import pytest

from web3_workers.dall_e_generate_descriptive_prompt import describe_image_with_openai_vision, \
    name_description_based_of_vision_description


# Test cases for different image categories
@pytest.mark.asyncio
@pytest.mark.parametrize("image_type,image_url,image_name,image_description,expected_success", [
    ("person", "https://pixelpact.s3.amazonaws.com/d3f4d743-9f53-471a-8d4e-83144427d562_11111.jpg", None, None, True),
    ("generated_art",
     "https://pixelpact.s3.us-east-2.amazonaws.com/generations/FS/01203f2a-648e-4909-adaa-a1c94dce28c3-FS-18271abb.png",
     None, None, True),
    ("art", "https://pixelpact.s3.us-east-2.amazonaws.com/arts/1.jpg", "Fight",
     "Art is always born as result of performancee", True)  # Assuming other is a valid type
])
async def test_describe_image_with_openai_vision_real(image_type, image_url, image_name, image_description,
                                                      expected_success):
    status, description = await describe_image_with_openai_vision(
        image_url, image_name, image_description, image_type)

    assert status == expected_success
    if expected_success:
        assert isinstance(description, str) and len(description) > 0


@pytest.mark.asyncio
async def test_generate_name_description_based_of_vision_description():
    image_type = 'generated_art'
    vision_description = """
    The artwork presented is an abstract painting rich in visual elements and textural variation.
    Dominated by a grayscale background, there are aggressive overtones of whites, pinks, yellows, reds,
    and blues that intersect and overlay the canvas. It appears as though the paint was applied with
    energetic gestures, perhaps throwing or dripping paint onto the surface, suggesting the chaos and vigor
    consistent with the title "Fight."

    The canvas bears the marks of impasto, where the paint is applied in such a thick manner that it adds a
    three-dimensional element to the work. The dense application in areas provides a tactile quality which
    evokes a sense of turmoil and intensity. There are evident drips and splatters, constructions that may
    emulate the chaotic nature of a struggle or conflict.

    Light and shadow are not depicted in a traditional manner, rather conveyed through the contrasts of color
    and the density of applied materials. Dark patches could signify depth or voids, while the vivid streaks
    of color give the impression of dynamic movement. There is no discernible pattern but rather an apparent
    randomness suggesting spontaneity and impulsiveness.

    Overall, the abstract nature and the thematic title combine to create an artwork that speaks more through
    emotion and sensory elements rather than representation. The piece can be perceived as a physical manifestation 
    of a battle, not in a literal sense, but rather as an expression of internal or conceptual conflict. The use of
    diverse materials and the absence of figurative content invite the viewer to experience the painting through their
    interpretation of the thematic concept of 'Fight'.
    
    """
    status, description = await name_description_based_of_vision_description(image_type, vision_description)

    assert status
    assert isinstance(description, str) and len(description) > 0


if __name__ == '__main__':
    pytest.main()
